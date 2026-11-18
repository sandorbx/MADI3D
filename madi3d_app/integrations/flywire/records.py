"""FlyWire identity and input observations, independent of scene and search state."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


FAFB_FRAME = "FlyWire FAFB14.1 / public m783"
MANC_FRAME = "MANC"
MCNS_FRAME = "Male CNS EM native coordinate space"
_UINT64_MAX = "18446744073709551615"


@dataclass(frozen=True)
class DatasetProfile:
    """Source contracts for exact releases; no scene or installation state.

    Product availability is independent: a public skeleton need not supply the
    radii required by SWC, and an annotation source does not imply a preview or
    a qualified output recipe. LOD availability is also checked per manifest.
    """

    dataset: str
    segmentation_version: str
    id_namespace: str
    coordinate_frame: str
    geometry_source: str
    geometry_revision: str
    mesh_directory: str
    mesh_lods: tuple[int, ...] | None
    skeleton_source: str
    skeleton_has_radii: bool
    annotation_source: str
    preview_template: str | None
    output_recipe_ids: tuple[str, ...]
    segmentation_uuid: str | None = None
    neuprint_dataset: str | None = None
    voxel_size_nm: tuple[int, int, int] | None = None
    mesh_format: str = "neuroglancer_multilod_draco"
    mesh_transform: tuple[float, ...] | None = None
    mesh_vertex_quantization_bits: int | None = None
    mesh_sharding: tuple[tuple[str, str | int], ...] | None = None
    decoded_nm_to_um: float = 0.001
    # Raw skeleton products may differ from the decoder's canonical nm contract.
    # MCNS supplies textual SWC XYZ and radii in units of 8 nm.
    skeleton_format: str = "neuroglancer_skeletons"
    skeleton_unit_nm: float = 1.0

    @property
    def representations(self) -> tuple[str, ...]:
        return ("mesh", "swc") if self.skeleton_has_radii else ("mesh",)


# Keep logical release, geometry product/revision and annotation product distinct.
# MANC's root info currently selects 1.2.3 properties; never follow that pointer
# to enrich an observation qualified as 1.2.1.
DATASET_PROFILES = (
    DatasetProfile(
        dataset="fafb", segmentation_version="783", id_namespace="root_id",
        coordinate_frame=FAFB_FRAME,
        geometry_source="https://storage.googleapis.com/flywire_v141_m783/mesh_mip_1_err_40",
        geometry_revision="783", mesh_directory="mesh_mip_1_err_40", mesh_lods=None,
        skeleton_source="https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783",
        skeleton_has_radii=True,
        annotation_source="https://storage.googleapis.com/flywire-data/codex/data/fafb/783/",
        preview_template="https://storage.googleapis.com/flywire-data/codex/skeleton_thumbnails/fafb/{root_id}.png",
        output_recipe_ids=("flywire-fafb14.1-um-to-jrc2018u-um-v1",),
    ),
    DatasetProfile(
        dataset="manc", segmentation_version="1.2.1", id_namespace="body_id",
        coordinate_frame=MANC_FRAME,
        geometry_source="https://storage.googleapis.com/manc-seg-v1p2/manc-seg-v1.2/mesh-multi-res",
        geometry_revision="7b5e8f7f805c4314bee37b75b4ff9292",
        mesh_directory="mesh-multi-res", mesh_lods=(0, 1, 2, 3),
        skeleton_source="https://storage.googleapis.com/manc-seg-v1p2/manc-seg-v1.2/skeleton",
        skeleton_has_radii=False,
        annotation_source="https://storage.googleapis.com/manc-seg-v1p2/manc-seg-v1.2/segment_properties_v1.2.1/info",
        preview_template="https://storage.googleapis.com/flywire-data/codex/skeleton_thumbnails/manc/{root_id}.png",
        output_recipe_ids=("manc-um-to-jrcvnc2018m-um-v1", "manc-um-to-jrcvnc2018u-um-v1"),
        segmentation_uuid="7b5e8f7f805c4314bee37b75b4ff9292",
        neuprint_dataset="manc:v1.2.1", voxel_size_nm=(8, 8, 8),
        mesh_transform=(16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0),
        mesh_vertex_quantization_bits=16,
        mesh_sharding=(("data_encoding", "gzip"), ("hash", "murmurhash3_x86_128"),
                       ("minishard_bits", 8), ("minishard_index_encoding", "gzip"),
                       ("preshift_bits", 6), ("shard_bits", 10)),
    ),
    DatasetProfile(
        dataset="mcns", segmentation_version="1.0", id_namespace="body_id",
        coordinate_frame=MCNS_FRAME,
        geometry_source="https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/meshes-malecns/multi-res-meshes",
        # A versioned upstream product, not a global immutable mesh/SWC UUID.
        geometry_revision="v1.0", mesh_directory="multi-res-meshes", mesh_lods=(0, 1, 2, 3),
        skeleton_source="https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc",
        skeleton_has_radii=True, skeleton_format="swc", skeleton_unit_nm=8.0,
        annotation_source="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        preview_template="https://storage.googleapis.com/flywire-data/codex/skeleton_thumbnails/mcns/{root_id}.png",
        output_recipe_ids=(
            "mcns-um-to-jrc2018m-um-v1", "mcns-um-to-jrc2018u-um-v1",
            "mcns-um-to-jrcvnc2018m-um-v1", "mcns-um-to-jrcvnc2018u-um-v1",
        ),
        neuprint_dataset="male-cns:v1.0", voxel_size_nm=(8, 8, 8),
        mesh_transform=(16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0),
        mesh_vertex_quantization_bits=16,
        mesh_sharding=(("data_encoding", "gzip"), ("hash", "murmurhash3_x86_128"),
                       ("minishard_bits", 8), ("minishard_index_encoding", "gzip"),
                       ("preshift_bits", 6), ("shard_bits", 10)),
    ),
)


def dataset_profile(dataset: str, segmentation_version: str) -> DatasetProfile:
    for profile in DATASET_PROFILES:
        if (dataset, segmentation_version) == (profile.dataset, profile.segmentation_version):
            return profile
    raise ValueError("Unsupported dataset/release; choose FAFB 783, MANC 1.2.1 or MCNS 1.0.")


def validate_root_id(value: str) -> str:
    """Accept exact positive uint64 decimal strings, never floating-point IDs."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError("Use a decimal root ID without a sign, decimal point or exponent.")
    if value.startswith("0"):
        raise ValueError("Root IDs must be positive decimal strings without leading zeros.")
    if len(value) > 20 or (len(value) == 20 and value > _UINT64_MAX):
        raise ValueError("Root ID exceeds the unsigned 64-bit range.")
    return value


@dataclass(frozen=True)
class ReleaseContext:
    """Explicit selection; a CSV or filename does not establish its release."""

    dataset: str
    segmentation_version: str
    origin: str = "user_selected"

    def __post_init__(self):
        dataset_profile(self.dataset, self.segmentation_version)
        if self.origin != "user_selected":
            raise ValueError("The initial release context must be explicitly user selected.")

    @property
    def profile(self) -> DatasetProfile:
        return dataset_profile(self.dataset, self.segmentation_version)


@dataclass(frozen=True)
class MetadataSourceReference:
    """Reference one input observation and row, without copying its entire table."""

    observation_id: str
    row_index: int

    def __post_init__(self):
        if not isinstance(self.observation_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.observation_id):
            raise ValueError("Metadata source requires a SHA-256 input observation identity.")
        if type(self.row_index) is not int or self.row_index < 1:
            raise ValueError("Metadata source row must be a positive integer.")


@dataclass(frozen=True)
class FlyWireRemoteDescriptor:
    """Versioned remote selection, suitable for existing object_metadata transport.

    Mesh LOD is nonnegative for upstream products; None denotes an unqualified local mesh.
    For upstream products: preferred before retrieval, actual on accepted geometry.
    Retrieval confirms the actual level before publishing an artifact. Units describe
    decoded canonical nm to output um, not raw skeleton product units or proof of
    retrieval. Product adapters first convert raw skeleton units to canonical nm.
    Cache paths, scene pose and scientific operation history do not belong here.
    """

    context: ReleaseContext
    root_id: str
    representation: str
    mesh_lod: int | None
    metadata_sources: tuple[MetadataSourceReference, ...] = ()
    schema_version: int = 1
    provider: str = "flywire"
    coordinate_frame: str | None = None
    source_units: str = "nm"
    output_units: str = "um"
    unit_scale: float = 0.001
    geometry_source: str | None = None
    geometry_revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.context, ReleaseContext):
            raise ValueError("A FlyWire descriptor requires an explicit release context.")
        validate_root_id(self.root_id)
        profile = self.context.profile
        if self.coordinate_frame is None:
            object.__setattr__(self, "coordinate_frame", profile.coordinate_frame)
        source = profile.skeleton_source if self.representation == "swc" else profile.geometry_source
        if profile.dataset != "fafb":
            for name, value in (("geometry_source", source), ("geometry_revision", profile.geometry_revision)):
                if getattr(self, name) is None:
                    object.__setattr__(self, name, value)
        if (self.geometry_source not in (None, source)
                or self.geometry_revision not in (None, profile.geometry_revision)):
            raise ValueError("Descriptor conflicts with its release's geometry product/revision.")
        if profile.dataset == "fafb" and (self.geometry_source is not None or self.geometry_revision is not None):
            raise ValueError("Schema-1 FAFB geometry products are specified by their release context.")
        if type(self.schema_version) is not int or self.schema_version != 1 or self.provider != "flywire":
            raise ValueError("Unsupported FlyWire descriptor schema/provider.")
        if self.representation == "mesh":
            if self.mesh_lod is not None and (type(self.mesh_lod) is not int or self.mesh_lod < 0):
                raise ValueError("A mesh requires its exact nonnegative LOD.")
            if self.mesh_lod is not None and profile.mesh_lods is not None and self.mesh_lod not in profile.mesh_lods:
                raise ValueError("This mesh LOD is not qualified for the selected release.")
        elif self.representation == "swc":
            if self.mesh_lod is not None:
                raise ValueError("Skeletons have no mesh LOD.")
        else:
            raise ValueError("Choose mesh or swc representation.")
        if self.representation not in profile.representations:
            raise ValueError("This source has no radius-bearing skeleton suitable for SWC; choose mesh.")
        if (self.coordinate_frame, self.source_units, self.output_units, self.unit_scale) != (
            profile.coordinate_frame, "nm", "um", profile.decoded_nm_to_um,
        ):
            raise ValueError("Descriptor conflicts with its release's native frame/unit contract.")
        if not isinstance(self.metadata_sources, tuple) or any(
            not isinstance(source, MetadataSourceReference) for source in self.metadata_sources
        ):
            raise ValueError("Metadata sources must be a tuple of input-row references.")

    @property
    def retrieval_identity(self) -> tuple[str, str, str, str, int | None]:
        return (self.context.dataset, self.context.segmentation_version,
                self.root_id, self.representation, self.mesh_lod)

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.context.dataset == "fafb":
            # Existing schema-1 FAFB descriptors keep their exact serialized shape.
            value.pop("geometry_source")
            value.pop("geometry_revision")
        value["metadata_sources"] = [asdict(source) for source in self.metadata_sources]
        return value

    @classmethod
    def from_dict(cls, value: dict) -> FlyWireRemoteDescriptor:
        """Validate at ingestion; do not coerce numeric IDs or unknown schemas."""
        try:
            fields = dict(value)
            if "schema_version" not in fields or "provider" not in fields:
                raise ValueError("A FlyWire remote descriptor requires its schema version and provider.")
            fields["context"] = ReleaseContext(**fields["context"])
            if fields["context"].dataset != "fafb" and any(
                fields.get(name) is None for name in ("coordinate_frame", "geometry_source", "geometry_revision")
            ):
                raise ValueError("A saved body-ID descriptor requires its frame and geometry product/revision.")
            fields["metadata_sources"] = tuple(
                MetadataSourceReference(**source) for source in fields.get("metadata_sources", ())
            )
            return cls(**fields)
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed FlyWire remote descriptor.") from exc


def remote_from_metadata(metadata):
    """Read the typed FlyWire namespace at project and scene boundaries."""
    if not metadata or "flywire_remote" not in metadata:
        return None
    return FlyWireRemoteDescriptor.from_dict(metadata["flywire_remote"])


@dataclass(frozen=True)
class InputObservation:
    """Bounded source observation shared by rows in one detached input batch."""

    observation_id: str
    input_kind: str
    source_name: str | None
    observed_at_utc: str
    headers: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputDiagnostic:
    code: str
    message: str
    row_index: int | None = None
    token: str | None = None
    column: str | None = None


@dataclass(frozen=True)
class InputRow:
    row_index: int
    cells: tuple[str, ...]
    root_id: str | None
    duplicate: bool = False


@dataclass(frozen=True)
class FlyWireInputs:
    context: ReleaseContext
    observation: InputObservation
    rows: tuple[InputRow, ...]
    diagnostics: tuple[InputDiagnostic, ...]
    recognized: bool = True
    # Detached catalog observations, separate from the original CSV/paste cells.
    # Accepted references freeze only their own rows and the consumed products.
    catalog: dict | None = None

    def __post_init__(self):
        if not isinstance(self.context, ReleaseContext):
            raise ValueError("Input review requires an explicit supported release context.")

    @property
    def retrieval_root_ids(self) -> tuple[str, ...]:
        return tuple(row.root_id for row in self.rows if row.root_id is not None and not row.duplicate)

    @property
    def valid_count(self) -> int:
        """Unique valid IDs; duplicates have their own count."""
        return len(self.retrieval_root_ids)

    @property
    def duplicate_count(self) -> int:
        return sum(row.duplicate for row in self.rows)

    @property
    def invalid_count(self) -> int:
        return sum(row.root_id is None for row in self.rows)

    def metadata(self, row: InputRow) -> dict[str, str]:
        """Exact source cells, including Query Syn ('Syn') and unknown columns.

        Structurally invalid rows remain available through row.cells/diagnostics;
        they cannot be projected unambiguously into named metadata.
        """
        headers = self.observation.headers
        if len(row.cells) != len(headers) or len({header.strip().casefold() for header in headers}) != len(headers):
            raise ValueError("This input row cannot be mapped unambiguously to its headers.")
        return dict(zip(headers, row.cells))
