"""Pinned contracts for the three supported downloadable local-search distributions.

This module is the single production owner of the normal local-search route's
downloadable distributions: which public object is downloaded, what it contains,
which search profile and geometry source it belongs to, and how its candidates
are identified. The complete available-library list is deterministic; importing
or listing it performs no network I/O, no GitHub Releases API lookup and no
Janelia S3 enumeration.

| MADI3D library ID    | Distribution            | Profile | Provider | Representation              |
| -------------------- | ----------------------- | ------- | -------- | --------------------------- |
| ``hemibrain-v1.2``   | FlyEM Hemibrain 1.2     | Brain   | Janelia  | dense TIFF + supplied gradient |
| ``flywire-fafb-v783``| FlyWire FAFB v783       | Brain   | MADI3D   | sparse RGB u32 release asset |
| ``flyem-manc-v1.2.1``| FlyEM MANC v1.2.1       | VNC     | MADI3D   | sparse RGB u32 release asset |

The Hemibrain distribution is the official Janelia
``Color_Depth_MIPs_For_Download`` archive (``BulkArchive``): it owns the archive
member directory that *is* the search library, the pinned candidate image
contract, the candidate filename grammar, and the explicit separation between a
*search collection* and the NeuronBridge *source library* consulted later for
geometry. The FlyWire FAFB and FlyEM MANC distributions are MADI3D-published
sparse RGB u32 ZIP assets on a pinned GitHub release (``SparseRelease``); their
provider-published SHA-256 is authoritative and no Releases API is consulted at
runtime. For the Janelia archive, whose multipart ETag is not a checksum, the
archive SHA-256 observed during installation becomes the installed-package
identity; there is no historical ETag/Last-Modified equality requirement on the
installed result.

It deliberately owns none of the following: downloads, extraction or search,
alignment evidence (``search_profiles``), or the installed-library readers
(``library_store``), which open an extracted distribution directly. The public
Online Lookup subsystem is separate and unchanged here.

The Hemibrain values were established by bounded direct inspection of the public
``janelia-flylight-color-depth`` bucket on 2026-10-30: ``HEAD`` for size, Range
support, ETag and Last-Modified; ZIP central-directory reads for member layout
and counts; and real members decoded through MADI3D's own RGB decode contract.
The public prefix holds exactly 18 objects and is not truncated. The survey is
reproducible through the opt-in probe in
``tests/test_neuronbridge_archive_catalog.py``.
"""
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from .cache import neuronbridge_cache_root
from .local_catalog import IMAGE_BUCKET, bulk_url
from .search_profiles import BRAIN, VNC, search_profile

ARCHIVE_PREFIX = "Color_Depth_MIPs_For_Download/"
ARCHIVE_URL_ROOT = f"https://s3.amazonaws.com/{IMAGE_BUCKET}/{ARCHIVE_PREFIX}"
# Official archives are extracted beneath the NeuronBridge downloaded-assets
# destination in their own directory so an archive library can never be mixed
# with an ad-hoc download.
ARCHIVE_LIBRARY_DIRECTORY = "ColorDepthLibraries"
SEARCH_COLLECTION_NAMESPACE = "janelia-cdm-archive"
CANDIDATE_DTYPE = "|u1"
CANDIDATE_PHOTOMETRIC = 2
CANDIDATE_COMPRESSION = 32773  # TIFF PackBits; the archive distribution's codec.
CANDIDATE_PAGES = 1
SURVEY_DATE = "2026-10-30"
SURVEY_PREFIX_OBJECT_COUNT = 18
CATALOG_DATA_VERSION = "v3_10_0"

# Each allowlisted collection is bound to its verified search contract. New
# collections require an explicit profile review and a searchable-image probe.
# This is the geometry-source allowlist the pinned descriptors cross-check.
VERIFIED_LIBRARIES = {"FlyEM_Hemibrain_v1.2.1": ("v1.2.1", "hemibrain:v1.2.1", BRAIN.profile_id),
                      "FlyWire_FAFB_v783_realign": ("v783", "flywire_fafb:v783", BRAIN.profile_id),
                      "FlyEM_MANC_v1.2.1": ("v1.2.1", "manc:v1.2.1", VNC.profile_id)}

# MADI3D-published sparse assets live on the public repository's pinned release.
# The URLs below are the exact distribution sources; runtime never asks the
# GitHub Releases API anything.
RELEASE_OWNER = "sandorbx/MADI3D"
RELEASE_TAG = "neuronbridge-sparse-assets-v1"
RELEASE_URL_ROOT = f"https://github.com/{RELEASE_OWNER}/releases/download/"
# The published asset format identifier. ``library_store.SparseLibraryReader``
# refuses any other ``index_format``, so the catalog and the reader share this
# one literal and the store layer stays importable from the catalog side.
SPARSE_RGB_U32_FORMAT = "positive-cds-sparse-rgb-u32-v1"

# The published sparse asset's own package layout, established on 2026-10-31 by
# bounded HTTP Range reads of both release assets' end-of-central-directory,
# central directory, ``package-manifest.json``, ``snapshot.json`` and
# ``inventory.sqlite``. The installer verifies against these facts and the
# runtime reader reads the same members; nothing here re-packs or re-indexes a
# package. ``metadata/`` exists only where the published package retains raw
# responses (MANC); FAFB embeds its evidence in the inventory instead, so it is
# deliberately not a required member.
PACKAGE_MANIFEST_FILENAME = "package-manifest.json"
PACKAGE_FORMAT = "madi3d-neuronbridge-sparse-assets"
PACKAGE_VERSION = 1
INVENTORY_FILENAME = "inventory.sqlite"
SHARD_DIRECTORY = "index"
SHARD_EXTENSION = ".npy"
# The inventory tables and columns a published package must provide: ``members``
# identifies each candidate and locates its pixel span, ``shards`` names the
# span's file. Other published tables are the producing store's own state and
# carry no meaning for an installed library.
PACKAGE_INVENTORY_TABLES = MappingProxyType({
    "members": ("ordinal", "member_key", "object_id", "nonzero_pixels",
                "shard", "pixel_offset", "pixel_count"),
    "shards": ("id", "sha256", "size"),
})
GRADIENT_INCLUDED = "included"
GRADIENT_GENERATED = "generated_on_demand"
GRADIENT_POLICIES = (GRADIENT_INCLUDED, GRADIENT_GENERATED)
PROVIDER_JANELIA = "Janelia"
PROVIDER_MADI3D = "MADI3D"

_WINDOWS_ILLEGAL_CHARS = '<>:"|?*'
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_MAX_MEMBER_BASENAME_BYTES = 255
_MAX_MEMBER_PATH_CHARACTERS = 240
_LIBRARY_ID_PATTERN = r"[a-z0-9][a-z0-9.-]*-v\d+(?:\.\d+)*"
_SHA256_PATTERN = r"[0-9a-f]{64}"


class UnsupportedArchive(ValueError):
    """Raised when a requested dense archive has no pinned official descriptor."""


class UnsupportedDistribution(ValueError):
    """Raised when a requested library ID is not one of the pinned distributions."""


@dataclass(frozen=True)
class CandidateIdentity:
    """Search membership parsed from one archive candidate member name."""
    archive_id: str
    body_id: str
    anatomical_name: str
    status: str
    variant: str  # "" for the base member; the archive's variant marker otherwise
    filename: str

    @property
    def is_variant(self):
        return bool(self.variant)


def portable_member_path(relative_path):
    """Return a ZIP member path that extracts identically on Windows and POSIX.

    Archive members become files under a user-writable library root, so a member
    that would escape that root, collide with a Windows device name or fail to
    round-trip through a POSIX path is rejected before anything is extracted.
    Apostrophes and parentheses are allowed: real Hemibrain members use them.
    """
    text = str(relative_path)
    path = PurePosixPath(text)
    if (not text or "\\" in text or text.startswith("/") or path.as_posix() != text
            or len(text) > _MAX_MEMBER_PATH_CHARACTERS):
        raise ValueError("Archive member path is not a normalized portable relative path.")
    if any(char in text for char in _WINDOWS_ILLEGAL_CHARS) or any(ord(char) < 32 for char in text):
        raise ValueError("Archive member path contains an unusable character.")
    parts = path.parts
    if not parts or text.endswith("/") or any(part == ".." for part in parts):
        raise ValueError("Archive member path must be a normalized file path.")
    basename = parts[-1]
    if len(basename.encode("utf-8")) > _MAX_MEMBER_BASENAME_BYTES:
        raise ValueError("Archive member basename exceeds a portable length.")
    if basename.rstrip(" .") != basename or basename.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError("Archive member name is not usable on Windows.")
    return text


def _compiled(pattern):
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError("Candidate filename grammar does not compile.") from exc
    if "body_id" not in compiled.groupindex:
        raise ValueError("Candidate filename grammar must name a body_id group.")
    return compiled


@dataclass(frozen=True)
class BulkArchive:
    """One official Janelia CDM archive with its pinned search and source identity."""

    library_id: str
    archive_id: str
    label: str
    provider: str
    representation: str
    gradient_policy: str
    expected_size_bytes: int
    provider_sha256: str
    expected_etag: str
    observed_last_modified: str
    profile_id: str
    library_release: str
    member_count: int
    candidate_directory: str
    candidate_extensions: tuple[str, ...]
    candidate_count: int
    candidate_original_count: int
    candidate_variant_count: int
    distinct_body_count: int
    candidate_canvas_yx: tuple[int, int]
    candidate_dtype: str
    candidate_photometric: int
    candidate_compression: int
    candidate_pages: int
    filename_pattern: str
    filename_parser: str
    variant_marker: str
    geometry_source_library: str
    geometry_release: str
    geometry_published_name_prefix: str
    membership_evidence: tuple[str, ...]
    companion_object_keys: tuple[str, ...] = ()
    verified_member_paths: tuple[str, ...] = ()
    non_candidate_member_paths: tuple[str, ...] = ()
    gradient_directory: str = ""
    shape_boundary_member: str | None = None
    _pattern: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not re.fullmatch(_LIBRARY_ID_PATTERN, self.library_id):
            raise ValueError("Distribution library ID requires a pinned lowercase versioned identifier.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,109}", self.archive_id):
            raise ValueError("Archive identifier is not usable as a portable directory name.")
        if self.provider not in (PROVIDER_JANELIA, PROVIDER_MADI3D):
            raise ValueError("Archive provider is not a supported distribution provider.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*-v\d+", self.representation):
            raise ValueError("Storage representation requires a versioned identifier.")
        if self.gradient_policy not in GRADIENT_POLICIES:
            raise ValueError("Gradient policy must be 'included' or 'generated_on_demand'.")
        if self.provider_sha256 and not re.fullmatch(_SHA256_PATTERN, self.provider_sha256):
            raise ValueError("Provider SHA-256 requires 64 lowercase hexadecimal digits.")
        bulk_url(self.url)
        profile = search_profile(self.profile_id)
        if self.candidate_canvas_yx != profile.search_canvas_yx:
            raise ValueError("Archive candidate canvas disagrees with its pinned search profile.")
        for name in (self.label, self.library_release, self.filename_parser):
            if not name or name != name.strip():
                raise ValueError("Archive descriptor requires a trimmed display value.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*-v\d+", self.filename_parser):
            raise ValueError("Candidate filename parser requires a versioned identifier.")
        if not self.candidate_extensions or any(
                extension != extension.lower() or extension not in (".tif", ".tiff")
                for extension in self.candidate_extensions):
            raise ValueError("Archive candidates must be TIFF members.")
        portable_member_path(self.candidate_directory)
        if self.candidate_directory.endswith(tuple(self.candidate_extensions)):
            raise ValueError("Archive candidate directory must be a directory.")
        if self.gradient_policy == GRADIENT_INCLUDED:
            if not self.gradient_directory:
                raise ValueError("A distribution that supplies gradients names its gradient directory.")
            portable_member_path(self.gradient_directory)
            if self.gradient_directory == self.candidate_directory:
                raise ValueError("The gradient directory must differ from the candidate directory.")
        elif self.gradient_directory:
            raise ValueError("Only a supplied-gradient distribution names a gradient directory.")
        if self.shape_boundary_member is not None:
            portable_member_path(self.shape_boundary_member)
            if (not self.gradient_directory
                    or not self.shape_boundary_member.startswith(self.gradient_directory + "/")
                    or not self.shape_boundary_member.endswith(".png")):
                raise ValueError("Shape boundary must be a PNG member inside the gradient directory.")
        counts = (self.member_count, self.candidate_count, self.candidate_original_count,
                  self.distinct_body_count, self.expected_size_bytes)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("Archive descriptor requires positive counts.")
        if (self.candidate_variant_count != self.candidate_count - self.candidate_original_count or
                self.candidate_variant_count < 0 or
                not self.candidate_original_count <= self.distinct_body_count <= self.candidate_count or
                self.member_count < self.candidate_count):
            raise ValueError("Archive candidate counts disagree with each other.")
        if bool(self.candidate_variant_count) != bool(self.variant_marker):
            raise ValueError("Archive variant marker requires variant candidates.")
        if (self.candidate_dtype != CANDIDATE_DTYPE or
                self.candidate_photometric != CANDIDATE_PHOTOMETRIC or
                self.candidate_compression != CANDIDATE_COMPRESSION or
                self.candidate_pages != CANDIDATE_PAGES):
            raise ValueError("Archive candidate image layout is outside the supported decoder contract.")
        if self.geometry_source_library:
            release, published, geometry_profile = VERIFIED_LIBRARIES.get(
                self.geometry_source_library, ("", "", ""))
            if (self.geometry_release != release or self.geometry_published_name_prefix != published
                    or geometry_profile != self.profile_id):
                raise ValueError("Geometry source library identity is not a verified NeuronBridge collection.")
        elif self.geometry_release or self.geometry_published_name_prefix:
            raise ValueError("Geometry release details require a source library.")
        if not self.membership_evidence or any(not note.strip() for note in self.membership_evidence):
            raise ValueError("Archive descriptor requires its survey evidence.")
        for key in self.companion_object_keys:
            portable_member_path(key)
            if not key.startswith(ARCHIVE_PREFIX) or not key.endswith(".zip"):
                raise ValueError("Archive companion must be another official archive object.")
        object.__setattr__(self, "_pattern", _compiled(self.filename_pattern))
        for path in self.verified_member_paths:
            if not self.accepts_member(path):
                raise ValueError("Pinned candidate member is outside the candidate directory.")
            self.parse_member(path)
        for path in self.non_candidate_member_paths:
            if self.accepts_member(path):
                raise ValueError("Pinned non-candidate member would be searched.")

    @property
    def object_key(self):
        return f"{ARCHIVE_PREFIX}{self.archive_id}.zip"

    @property
    def url(self):
        return ARCHIVE_URL_ROOT + self.archive_id + ".zip"

    @property
    def profile(self):
        return search_profile(self.profile_id)

    @property
    def alignment_space(self):
        return self.profile.alignment_space

    @property
    def search_collection_id(self):
        """Identity of what this archive makes searchable; never a geometry source."""
        return f"{SEARCH_COLLECTION_NAMESPACE}/{self.archive_id}/{self.candidate_directory}"

    @property
    def geometry_proven(self):
        return bool(self.geometry_source_library)

    @property
    def candidate_identity(self):
        """Versioned contract parsing one candidate member into its body ID."""
        return self.filename_parser

    @property
    def source_release(self):
        """The provider-side release this official archive is published as."""
        return self.library_release

    def accepts_member(self, member_path):
        """Return whether an archive member belongs to this archive's search library."""
        try:
            portable_member_path(member_path)
        except ValueError:
            return False
        prefix = self.candidate_directory + "/"
        if not member_path.startswith(prefix):
            return False
        return member_path.rsplit("/", 1)[1].lower().endswith(self.candidate_extensions)

    def parse_member(self, member_path):
        """Parse search membership; the body ID, not the filename, drives geometry later."""
        filename = portable_member_path(member_path).rsplit("/", 1)[-1]
        match = self._pattern.fullmatch(filename)
        if match is None:
            raise ValueError(f"Member name does not match the {self.filename_parser} grammar.")
        groups = match.groupdict()
        return CandidateIdentity(self.archive_id, groups["body_id"], groups.get("anatomical_name") or "",
                                 groups.get("status") or "", groups.get("variant") or "", filename)

    def supplied_gradient_member(self, member_path):
        """Archive-relative supplied-gradient entry for one candidate member.

        A supplied-gradient distribution names each gradient after the candidate
        filename stem inside its own gradient directory. ``None`` means this
        distribution publishes no gradient file at all, so the pinned producer
        generates one; a named entry may still be absent from an installed
        archive when a single candidate lost its gradient.
        """
        if self.gradient_policy != GRADIENT_INCLUDED:
            return None
        filename = portable_member_path(member_path).rsplit("/", 1)[-1]
        return f"{self.gradient_directory}/{filename.rsplit('.', 1)[0]}.png"

    def to_dict(self):
        values = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name != "_pattern"}
        values.update(object_key=self.object_key, url=self.url, alignment_space=self.alignment_space,
                      search_collection_id=self.search_collection_id, geometry_proven=self.geometry_proven)
        return values


HEMIBRAIN_ARCHIVE = BulkArchive(
    library_id="hemibrain-v1.2",
    archive_id="EM_Hemibrain_Ver1.2_CDM_PackBits_gradient",
    label="FlyEM Hemibrain 1.2",
    provider=PROVIDER_JANELIA,
    representation="dense-tiff-packbits-rgb-v1",
    gradient_policy=GRADIENT_INCLUDED,  # The same official archive ships every candidate gradient.
    expected_size_bytes=887_701_869,
    # The Janelia bucket publishes no whole-object SHA-256 for this multipart
    # object; the archive SHA-256 observed during installation is the identity.
    provider_sha256="",
    expected_etag="a19c3752661b44f8f46d2248215b5bed-52",
    observed_last_modified="Thu, 01 Jul 2021 21:17:58 GMT",
    profile_id="nb02-prepared-signal-v1",
    library_release="v1.2",
    member_count=88_956,
    candidate_directory="EM_Hemibrain1.2_CDM",
    candidate_extensions=(".tif",),
    candidate_count=44_476,
    candidate_original_count=30_789,
    candidate_variant_count=13_687,
    distinct_body_count=30_789,
    candidate_canvas_yx=(566, 1210),
    candidate_dtype=CANDIDATE_DTYPE,
    candidate_photometric=CANDIDATE_PHOTOMETRIC,
    candidate_compression=CANDIDATE_COMPRESSION,
    candidate_pages=CANDIDATE_PAGES,
    filename_pattern=r"(?P<body_id>\d+)-(?P<anatomical_name>.*)-(?P<status>[A-Za-z'\-]+)_18U(?P<variant>_FL)?\.tif",
    filename_parser="hemibrain-cdm-18u-v1",
    variant_marker="_FL",
    geometry_source_library="FlyEM_Hemibrain_v1.2.1",
    geometry_release="v1.2.1",
    geometry_published_name_prefix="hemibrain:v1.2.1",
    membership_evidence=(
        f"{SURVEY_DATE} ZIP central directory: 88956 members; 44476 TIFF candidates in "
        "EM_Hemibrain1.2_CDM/ (30789 original plus 13687 _FL) over 30789 distinct body IDs; "
        "44477 PNGs (44476 per-candidate gradients plus one Hemibrain boundary), 1 search-plugin JAR "
        "and 2 JVM crash logs are not candidates.",
        "Every _FL member has the exact original filename in the same directory apart from the "
        "trailing _FL marker, and its pixels differ from the original member, so both are "
        "separate search members of one body ID.",
        f"NeuronBridge {CATALOG_DATA_VERSION} declares FlyEM_Hemibrain_v1.2.1 at 44477 searchable "
        "images (30790 originals plus 13687 -FL). Every archive body ID exists in that collection "
        "and body 2307027729 exists only in NeuronBridge, so this archive is a 44476-member search "
        "collection, not the NeuronBridge collection.",
        "The Janelia bucket README lists FlyEM_Hemibrain_v1.2.1 at 30790 images, which counts only "
        "the original CDMs and must not be presented as this archive's candidate count.",
        "Companion EM_Hemibrain1.2_SWC.zip holds 30789 SWC members whose stems equal the original "
        "CDM candidate stems one to one; its units were not established by this survey.",
    ),
    companion_object_keys=(f"{ARCHIVE_PREFIX}EM_Hemibrain1.2_SWC.zip",
                           f"{ARCHIVE_PREFIX}EM_hemibrain_ver1.2_OBJ_JRC2018U.zip"),
    verified_member_paths=(
        "EM_Hemibrain1.2_CDM/1001453586-KCa'b'-ap1-RT_18U.tif",
        "EM_Hemibrain1.2_CDM/1002360103-AVLP464-RT_18U.tif",
        "EM_Hemibrain1.2_CDM/1002507131-PEN_a(PEN1)-RT_18U.tif",
        "EM_Hemibrain1.2_CDM/1002507131-PEN_a(PEN1)-RT_18U_FL.tif",
    ),
    non_candidate_member_paths=(
        "EM_Hemibrain1.2_CDM/hs_err_pid13072.log",
        "EM_Hemibrain1.2_gradient/1002507131-PEN_a(PEN1)-RT_18U.png",
        "EM_MIP_Mask_Search.jar",
    ),
    # 2026-10-31 central-directory read: EM_Hemibrain1.2_gradient/ holds 44477
    # PNGs: 44476 per-candidate gradients (original and _FL) plus the boundary.
    gradient_directory="EM_Hemibrain1.2_gradient",
    shape_boundary_member="EM_Hemibrain1.2_gradient/MAX_hemi_to_JRC2018U_fineTune.png",
)

@dataclass(frozen=True)
class SparseRelease:
    """One MADI3D-published sparse RGB u32 ZIP asset on the pinned GitHub release."""

    library_id: str
    label: str
    provider: str
    release_tag: str
    asset_name: str
    expected_size_bytes: int
    provider_sha256: str
    profile_id: str
    candidate_count: int
    representation: str
    candidate_identity: str
    gradient_policy: str
    geometry_source_library: str
    geometry_release: str
    geometry_published_name_prefix: str

    def __post_init__(self):
        if not re.fullmatch(_LIBRARY_ID_PATTERN, self.library_id):
            raise ValueError("Distribution library ID requires a pinned lowercase versioned identifier.")
        if self.provider != PROVIDER_MADI3D:
            raise ValueError("Sparse local-search distributions are published by MADI3D.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.release_tag):
            raise ValueError("Sparse distribution release tag is not a usable pinned tag.")
        portable_member_path(self.asset_name)
        if not self.asset_name.endswith(".zip"):
            raise ValueError("Sparse distribution asset must be a ZIP archive.")
        if not re.fullmatch(_SHA256_PATTERN, self.provider_sha256):
            raise ValueError("Sparse distribution requires its authoritative provider SHA-256.")
        profile = search_profile(self.profile_id)
        if type(self.expected_size_bytes) is not int or self.expected_size_bytes <= 0:
            raise ValueError("Sparse distribution requires its pinned compressed size.")
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            raise ValueError("Sparse distribution requires a positive candidate count.")
        for name in (self.label, self.representation, self.candidate_identity):
            if not name or name != name.strip():
                raise ValueError("Sparse distribution descriptor requires a trimmed display value.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*-v\d+", self.representation):
            raise ValueError("Storage representation requires a versioned identifier.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*-v\d+", self.candidate_identity):
            raise ValueError("Candidate identity contract requires a versioned identifier.")
        if self.gradient_policy not in GRADIENT_POLICIES:
            raise ValueError("Gradient policy must be 'included' or 'generated_on_demand'.")
        release, published, profile_id = VERIFIED_LIBRARIES.get(
            self.geometry_source_library, ("", "", ""))
        if (self.geometry_release != release or self.geometry_published_name_prefix != published
                or profile_id != self.profile_id):
            raise ValueError("Geometry source library identity is not a verified NeuronBridge collection.")
        if profile.anatomical_area not in ("Brain", "VNC"):
            raise ValueError("Sparse distribution profile is outside the supported search spaces.")

    @property
    def url(self):
        return f"{RELEASE_URL_ROOT}{self.release_tag}/{self.asset_name}"

    @property
    def profile(self):
        return search_profile(self.profile_id)

    @property
    def alignment_space(self):
        return self.profile.alignment_space

    @property
    def source_release(self):
        """The pinned GitHub release this asset is downloaded from."""
        return self.release_tag

    def to_dict(self):
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(url=self.url, alignment_space=self.alignment_space)
        return values


FLYWIRE_FAFB_DISTRIBUTION = SparseRelease(
    library_id="flywire-fafb-v783",
    label="FlyWire FAFB v783",
    provider=PROVIDER_MADI3D,
    release_tag=RELEASE_TAG,
    asset_name="FlyWire_FAFB_v783_realign-sparse-v1.zip",
    expected_size_bytes=1_245_037_218,
    provider_sha256="54d036a77fbcd82d7f25a9758ee5d4271033ad2a47b698fc799a76b891e402aa",
    profile_id=BRAIN.profile_id,
    candidate_count=144_005,
    representation=SPARSE_RGB_U32_FORMAT,
    candidate_identity="numeric-neuron-body-id-v1",
    gradient_policy=GRADIENT_GENERATED,
    geometry_source_library="FlyWire_FAFB_v783_realign",
    geometry_release="v783",
    geometry_published_name_prefix="flywire_fafb:v783",
)

FLYEM_MANC_DISTRIBUTION = SparseRelease(
    library_id="flyem-manc-v1.2.1",
    label="FlyEM MANC v1.2.1",
    provider=PROVIDER_MADI3D,
    release_tag=RELEASE_TAG,
    asset_name="FlyEM_MANC_v1.2.1-sparse-v1.zip",
    expected_size_bytes=1_091_621_708,
    provider_sha256="e5f7011954b333a97a937771a15064b52289a46bdcfe0bcadc5b124751e995ba",
    profile_id=VNC.profile_id,
    candidate_count=24_350,
    representation=SPARSE_RGB_U32_FORMAT,
    candidate_identity="numeric-neuron-body-id-v1",
    gradient_policy=GRADIENT_GENERATED,
    geometry_source_library="FlyEM_MANC_v1.2.1",
    geometry_release="v1.2.1",
    geometry_published_name_prefix="manc:v1.2.1",
)

# The complete available-library list. Declared order is presentation order.
LOCAL_SEARCH_DISTRIBUTIONS = MappingProxyType({
    descriptor.library_id: descriptor
    for descriptor in (HEMIBRAIN_ARCHIVE, FLYWIRE_FAFB_DISTRIBUTION, FLYEM_MANC_DISTRIBUTION)
})
# Dense official Janelia archives reachable through the archive installer route.
ARCHIVE_CATALOG = MappingProxyType({descriptor.archive_id: descriptor
                                    for descriptor in LOCAL_SEARCH_DISTRIBUTIONS.values()
                                    if isinstance(descriptor, BulkArchive)})


def distributions():
    """Return the three pinned downloadable local-search distributions, in order."""
    return tuple(LOCAL_SEARCH_DISTRIBUTIONS.values())


def distribution(library_id):
    """Return the distribution for one stable MADI3D library ID."""
    descriptor = LOCAL_SEARCH_DISTRIBUTIONS.get(library_id)
    if descriptor is None:
        raise UnsupportedDistribution(
            "No pinned local-search distribution uses that MADI3D library ID. The supported "
            "distributions are " + ", ".join(LOCAL_SEARCH_DISTRIBUTIONS) + ".")
    return descriptor


def archives():
    """Return every dense official Janelia archive with a pinned descriptor."""
    return tuple(ARCHIVE_CATALOG.values())


def archive(archive_id):
    descriptor = ARCHIVE_CATALOG.get(archive_id)
    if descriptor is None:
        raise UnsupportedArchive("No official Janelia archive is pinned for that identifier.")
    return descriptor


def candidate_identity(archive_id, member_path):
    """Parse one candidate member of a proven archive."""
    descriptor = archive(archive_id)
    if not descriptor.accepts_member(member_path):
        raise UnsupportedArchive("Member is outside the pinned candidate directory of that archive.")
    return descriptor.parse_member(member_path)


def archive_library_directory(archive_id, root=None):
    """Return the dedicated extraction directory for one official archive.

    The archive identifier is validated as a portable directory name here; the
    descriptor itself is checked by the calling installer boundary, so a test
    descriptor injected there still resolves a safe managed directory.
    """
    if not isinstance(archive_id, str):
        raise ValueError("Archive identifier must be a portable directory name.")
    portable_member_path(archive_id)
    base = Path(root) if root is not None else neuronbridge_cache_root()
    return Path(base) / ARCHIVE_LIBRARY_DIRECTORY / archive_id
