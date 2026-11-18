"""Pinned NeuronBridge search contracts, independent of UI and source naming.

Aliases describe upstream terminology; only a checksum and exact working grid
prove template placement. See docs/neuronbridge-vnc.md for reference evidence.
"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class EquivalentTemplateGrid:
    """Checksum-pinned input sampling in the profile's physical coordinates."""
    template_id: str
    template_sha256: str
    template_shape_zyx: tuple[int, int, int]
    spacing_xyz_um: tuple[float, float, float]
    template_url: str
    origin_xyz_um: tuple[float, float, float] = (0., 0., 0.)
    direction: tuple[tuple[float, ...], ...] = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


@dataclass(frozen=True)
class SearchProfile:
    profile_id: str
    anatomical_area: str
    alignment_space: str
    aliases: tuple[str, ...]
    template_sha256: str
    template_shape_zyx: tuple[int, int, int]
    spacing_xyz_um: tuple[float, float, float]
    search_canvas_yx: tuple[int, int]
    content_offset_yx: tuple[int, int]
    generator: str
    template_url: str
    equivalent_grids: tuple[EquivalentTemplateGrid, ...] = ()
    origin_xyz_um: tuple[float, float, float] = (0., 0., 0.)
    direction: tuple[tuple[float, ...], ...] = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    projection_axis: str = "Z"
    depth_axis: str = "Z"
    lut_sha256: str = "10bdeb45546b89d5c311c423975072ea11a5d32ea6eb18c2b950583605a7389f"
    # Upstream mipsearch.js uses the same two rectangles for both areas.
    # (anchor, x, y, width, height), with x measured from the named edge.
    score_exclusions: tuple[tuple, ...] = (("left", 0, 0, 330, 100), ("right", 0, 0, 250, 90))
    warnings: tuple[str, ...] = (
        "Template origin (0,0,0) micrometres is a working assumption; the source header omits origin.",
        "Template coordinate basis is unnamed; anatomical axis directions are unverified.",
    )

    @property
    def template_id(self):
        return self.alignment_space

    @property
    def label(self):
        return f"{self.anatomical_area} · {self.aliases[0]}"


BRAIN = SearchProfile(
    "nb02-prepared-signal-v1", "Brain", "JRC2018_Unisex_20x_HR", ("JRC2018U_HR",),
    "00b8fe7db7981523d28af9ee629cc614515d7c95c6d3ffb6293540c369741a70",
    (174, 566, 1210), (0.5189161, 0.5189161, 1.0), (566, 1210), (0, 0), "madi3d-brain-cdm/1",
    "https://janelia-flylight-templates.s3.amazonaws.com/JRC2018_Unisex_20x_HR/JRC2018_UNISEX_20x_HR.nrrd",
)
VNC = SearchProfile(
    "nb-vnc-prepared-signal-v1", "VNC", "JRC2018_VNC_Unisex_40x_DS",
    ("JRC2018VNCU_HR", "JRC2018_VNC_UNISEX_461"),
    "14935d7d287dd24e93d0bd83238891436479ab2b98c87da3ccecd24fbda4419f",
    (219, 1119, 573), (0.461122, 0.461122, 0.7), (1209, 573), (90, 0), "madi3d-vnc-cdm/1",
    "https://janelia-flylight-templates.s3.amazonaws.com/JRC2018_VNC_Unisex/JRC2018_VNC_UNISEX_461.nrrd",
    equivalent_grids=(EquivalentTemplateGrid(
        "JRCVNC2018U", "7a58f0924b62c4a90a0e9106a029fba4fc0b112b66591ce2e577023eae23c3e5",
        (382, 1290, 660), (0.4, 0.4, 0.4),
        "https://ndownloader.figshare.com/files/12809834?private_link=a677185a43121977ce61",
    ),),
)
SEARCH_PROFILES = MappingProxyType({p.profile_id: p for p in (BRAIN, VNC)})
BRAIN_ISOTROPIC_DIMENSIONS_XYZ = (1652, 773, 456)


def dimension_profile(dimensions_xyz):
    """Suggest a target from an exact volume shape; this is not alignment evidence."""
    dimensions = tuple(dimensions_xyz)
    matches = [p for p in SEARCH_PROFILES.values()
               if dimensions == p.template_shape_zyx[::-1]
               or any(dimensions == grid.template_shape_zyx[::-1] for grid in p.equivalent_grids)
               or (p == BRAIN and dimensions == BRAIN_ISOTROPIC_DIMENSIONS_XYZ)]
    return matches[0] if len(matches) == 1 else None


def search_profile(profile_id):
    try:
        return SEARCH_PROFILES[profile_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown NeuronBridge search profile.") from exc


def query_profile(record):
    """Profile of a retained query record; generated identity is authoritative.

    Imported bare pixels remain unverified: for a bare import a canvas selects
    only the numerical search contract. It never supplies mapping evidence,
    calibration or template identity.
    """
    # Deferred: cdm_records imports this module, and query classification must
    # stay here at the profile owner rather than in transport or catalog code.
    from .cdm_records import CDMArtifact, ImportedCDMImage
    from .query import ROI_QUERY_VERSION
    if record["version"] == ROI_QUERY_VERSION:
        profile = record["derivation"]["profile"]
        return search_profile(profile) if profile is not None else None
    if record["version"] == 2:
        image = ImportedCDMImage.from_dict(record["imported_image"])
        artifact = image.generation_artifact
        if artifact is None:
            return next((p for p in SEARCH_PROFILES.values()
                         if image.shape_yx_rgb == (*p.search_canvas_yx, 3)), None)
    else:
        artifact = CDMArtifact.from_dict(record["artifact"])
    return search_profile(artifact.profile)


def compatible_query(record, library=None):
    """Whether a retained query may search the selected library or profile.

    ``library`` is ``None`` (any supported profile), a profile ID, a pinned
    distribution descriptor (``profile_id``), or a retained-snapshot ``Library``
    (``profile``). Brain and VNC searches never combine.
    """
    profile = query_profile(record)
    if library is None:
        return profile is not None
    if profile is None:
        return False
    declared = getattr(library, "profile_id", None) or getattr(library, "profile", None) or library
    return profile.profile_id == declared


def template_profile(template_id, checksum):
    for profile in SEARCH_PROFILES.values():
        if template_id == profile.template_id and checksum == profile.template_sha256:
            return profile
    raise ValueError("Template identity/checksum does not match a pinned NeuronBridge search profile.")
