"""Pinned NeuronBridge search contracts, independent of UI and source naming.

Aliases describe upstream terminology; only a checksum and exact working grid
prove template placement. See docs/neuronbridge-vnc.md for reference evidence.
"""
from dataclasses import dataclass
from types import MappingProxyType


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
)
SEARCH_PROFILES = MappingProxyType({p.profile_id: p for p in (BRAIN, VNC)})


def search_profile(profile_id):
    try:
        return SEARCH_PROFILES[profile_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown NeuronBridge search profile.") from exc


def template_profile(template_id, checksum):
    for profile in SEARCH_PROFILES.values():
        if template_id == profile.template_id and checksum == profile.template_sha256:
            return profile
    raise ValueError("Template identity/checksum does not match a pinned NeuronBridge search profile.")
