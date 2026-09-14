"""Immutable generation evidence for pinned NeuronBridge search profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json

import numpy as np

from madi3d_app.volume.geometry import VolumeWorkingGrid, invertible_affine4
from .records import ChannelSelection, SourceIdentity
from .search_profiles import BRAIN, SEARCH_PROFILES, search_profile, template_profile

# Existing public brain constants remain stable for saved schema-1 artifacts.
PROFILE = BRAIN.profile_id
GENERATOR = BRAIN.generator
TEMPLATE_ID = BRAIN.template_id
TEMPLATE_SHA256 = BRAIN.template_sha256
TEMPLATE_SHAPE = BRAIN.template_shape_zyx
LUT_SHA256 = BRAIN.lut_sha256
TEMPLATE_WARNINGS = BRAIN.warnings


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit and nonempty.")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256 checksum.")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}.")


def _affine(value):
    matrix = invertible_affine4(value)
    if not np.array_equal(matrix[3], (0, 0, 0, 1)):
        raise ValueError("Mapping requires an exact homogeneous affine last row.")
    return tuple(tuple(float(v) for v in row) for row in matrix)


@dataclass(frozen=True)
class CDMSelection:
    """Identity of ONE already selected scalar ZYX array, including a static frame 0."""

    source_id: str
    source_revision: str
    source: SourceIdentity
    channel: ChannelSelection
    frame_index: int
    axis_order: str
    preparation_json: str | None = None

    def __post_init__(self):
        _text(self.source_id, "source_id")
        _text(self.source_revision, "source_revision")
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.channel, ChannelSelection):
            raise ValueError("Explicit source and channel records are required.")
        if self.source.channel is not None and self.source.channel != self.channel:
            raise ValueError("Source and selected channel disagree.")
        _integer(self.frame_index, "frame_index", 0, 2**63 - 1)
        if self.axis_order != "ZYX":
            raise ValueError("Resolve axis ambiguity and supply one explicit ZYX scalar frame.")
        if self.preparation_json is not None:
            prepared = json.loads(self.preparation_json)
            if not isinstance(prepared, dict):
                raise ValueError("Input preparation evidence must be an object.")
            object.__setattr__(self, "preparation_json", canonical_json(prepared))

    def to_dict(self):
        data = asdict(self)
        if self.preparation_json is None:
            data.pop("preparation_json")
        return data

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["source"] = SourceIdentity.from_dict(data["source"])
        data["channel"] = ChannelSelection(**data["channel"])
        return cls(**data)


@dataclass(frozen=True)
class CDMGeometry:
    """Frozen authoritative grid and pose evidence; never implicitly reapplied."""

    working_grid: VolumeWorkingGrid
    revision: str
    pose: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if not isinstance(self.working_grid, VolumeWorkingGrid):
            raise ValueError("A frozen VolumeWorkingGrid is required.")
        _text(self.revision, "geometry revision")
        object.__setattr__(self, "pose", _affine(self.pose))

    def to_dict(self):
        return {"working_grid": self.working_grid.to_dict(), "revision": self.revision,
                "pose": self.pose}

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["working_grid"] = VolumeWorkingGrid.from_dict(data["working_grid"])
        return cls(**data)


@dataclass(frozen=True)
class CDMMapping:
    """The COMPLETE source XYZ index -> template XYZ index affine consumed once.

    Evidence is an external registration/index-placement record ID and checksum,
    not the name of a preset. The generator does not perform registration QC.
    """

    source_index_to_template_index: tuple[tuple[float, ...], ...]
    revision: str
    evidence_id: str | None = None
    evidence_sha256: str | None = None
    template_id: str = TEMPLATE_ID
    template_sha256: str = TEMPLATE_SHA256
    evidence_json: str | None = None
    assumptions: tuple[str, ...] = ()
    placement: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "source_index_to_template_index", _affine(self.source_index_to_template_index))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        for assumption in self.assumptions:
            _text(assumption, "mapping assumption")
        if self.placement not in (None, "automatic", "isotropic", "working"):
            raise ValueError("Unknown exploratory placement.")
        _text(self.revision, "mapping revision")
        template_profile(self.template_id, self.template_sha256)
        if (self.evidence_id is None) != (self.evidence_sha256 is None):
            raise ValueError("Mapping evidence needs both an ID and checksum.")
        if self.evidence_id is not None:
            _text(self.evidence_id, "mapping evidence ID")
            _sha(self.evidence_sha256)
        if self.evidence_json is not None:
            encoded = canonical_json(json.loads(self.evidence_json)).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != self.evidence_sha256:
                raise ValueError("Mapping evidence checksum mismatch.")


@dataclass(frozen=True)
class CDMParameters:
    mode: str = "signal"
    display_range: tuple[int, int] | None = None
    threshold: int = 0
    depth_range: tuple[int, int] | None = None
    interpolation: str = "exact"
    allow_exploratory: bool = False
    scalar_range: tuple[float, float] | None = None
    profile: str = PROFILE

    def __post_init__(self):
        if self.scalar_range is not None:
            values = np.asarray(self.scalar_range, dtype=float)
            if (values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]
                    or not np.isfinite(float(values[1]) - float(values[0]))):
                raise ValueError("Scalar conversion needs two increasing finite endpoints.")
            if self.display_range is not None or self.mode == "binary_mask":
                raise ValueError("Scalar conversion cannot be combined with uint16 conversion or binary occupancy.")
            object.__setattr__(self, "scalar_range", tuple(float(v) for v in values))
        if self.mode not in ("signal", "signal_mask", "binary_mask"):
            raise ValueError("Unknown prepared input mode.")
        if self.display_range is not None:
            if len(self.display_range) != 2:
                raise ValueError("display_range needs two endpoints.")
            lo, hi = self.display_range
            _integer(lo, "display minimum", 0, 65534)
            _integer(hi, "display maximum", lo + 1, 65535)
            object.__setattr__(self, "display_range", (lo, hi))
        _integer(self.threshold, "threshold", 0, 255)
        last_depth = search_profile(self.profile).template_shape_zyx[0] - 1
        if self.depth_range is None:
            object.__setattr__(self, "depth_range", (0, last_depth))
        if len(self.depth_range) != 2:
            raise ValueError("depth_range needs two inclusive global indices.")
        start, end = self.depth_range
        _integer(start, "first depth", 0, last_depth)
        _integer(end, "last depth", start, last_depth)
        object.__setattr__(self, "depth_range", (start, end))
        if self.interpolation not in ("exact", "nearest"):
            raise ValueError("Use exact index placement or explicit nearest interpolation.")
        if type(self.allow_exploratory) is not bool:
            raise ValueError("allow_exploratory must be boolean.")


@dataclass(frozen=True)
class CDMArtifact:
    """Brain schema 1 / VNC schema 2; no mutable containers or image buffers.

    Completed means pixels were generated. Compatibility remains a separate,
    unperformed validation; mapping evidence alone is not a QC pass.
    """

    selection: CDMSelection
    geometry: CDMGeometry
    mapping: CDMMapping
    parameters: CDMParameters
    source_dtype: str | None
    source_pixel_sha256: str | None
    mask_selection: CDMSelection | None
    mask_revision: str | None
    mask_dtype: str | None
    mask_pixel_sha256: str | None
    rgb_pixel_sha256: str
    warnings: tuple[str, ...]
    numpy_version: str
    scipy_version: str | None
    png_file_sha256: str | None = None
    png_encoder: str | None = None
    schema_version: int = 1
    profile: str = PROFILE
    generator: str = GENERATOR
    lut_sha256: str = LUT_SHA256
    template_id: str = TEMPLATE_ID
    template_sha256: str = TEMPLATE_SHA256
    template_shape_zyx: tuple[int, int, int] = TEMPLATE_SHAPE
    template_spacing_xyz_um: tuple[float, float, float] = (0.5189161, 0.5189161, 1.0)
    template_working_origin_xyz_um: tuple[float, float, float] = (0.0, 0.0, 0.0)
    template_direction: tuple[tuple[float, ...], ...] = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    execution_status: str = "completed"
    compatibility_status: str = "not_validated"
    search_canvas_shape_yx: tuple[int, int] | None = None
    content_offset_yx: tuple[int, int] | None = None
    anatomical_area: str | None = None
    alignment_space: str | None = None

    def __post_init__(self):
        for name, kind in (("selection", CDMSelection), ("geometry", CDMGeometry),
                           ("mapping", CDMMapping), ("parameters", CDMParameters)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be {kind.__name__}.")
        profile = search_profile(self.profile)
        version = 1 if profile == BRAIN else 2
        _integer(self.schema_version, "schema_version", version, version)
        if (template_profile(self.mapping.template_id, self.mapping.template_sha256) != profile
                or self.parameters.profile != self.profile):
            raise ValueError("CDM mapping, parameters and artifact search profiles disagree.")
        for name, expected in (("search_canvas_shape_yx", profile.search_canvas_yx),
                               ("content_offset_yx", profile.content_offset_yx),
                               ("anatomical_area", profile.anatomical_area),
                               ("alignment_space", profile.alignment_space)):
            value = getattr(self, name)
            if version == 1:
                if value is not None:
                    raise ValueError("Schema-1 brain artifacts retain their original layout contract.")
            else:
                if isinstance(expected, tuple) and value is not None:
                    value = tuple(value)
                if value != expected:
                    raise ValueError(f"Unsupported CDM {name}.")
                object.__setattr__(self, name, value)
        _text(self.numpy_version, "NumPy version")
        if self.parameters.interpolation == "nearest":
            _text(self.scipy_version, "SciPy version")
        elif self.scipy_version is not None:
            raise ValueError("Exact placement does not consume SciPy.")
        for name, expected in (("template_shape_zyx", profile.template_shape_zyx),
                               ("template_spacing_xyz_um", profile.spacing_xyz_um),
                               ("template_working_origin_xyz_um", profile.origin_xyz_um),
                               ("template_direction", profile.direction)):
            value = getattr(self, name)
            frozen = tuple(tuple(row) for row in value) if name == "template_direction" else tuple(value)
            if frozen != expected:
                raise ValueError(f"Unsupported {name}.")
            object.__setattr__(self, name, frozen)
        for name, expected in (("schema_version", version), ("profile", profile.profile_id), ("generator", profile.generator),
                               ("lut_sha256", profile.lut_sha256), ("template_id", profile.template_id),
                               ("template_sha256", profile.template_sha256), ("execution_status", "completed"),
                               ("compatibility_status", "not_validated")):
            if getattr(self, name) != expected:
                raise ValueError(f"Unsupported CDM {name}.")
        for value in (self.source_pixel_sha256, self.mask_pixel_sha256, self.rgb_pixel_sha256,
                      self.png_file_sha256):
            if value is not None:
                _sha(value)
        if self.rgb_pixel_sha256 is None:
            raise ValueError("Completed artifact needs a pixel checksum.")
        binary = self.parameters.mode == "binary_mask"
        if binary != (self.source_pixel_sha256 is None) or binary != (self.source_dtype is None):
            raise ValueError("Signal checksum and dtype must match the declared input mode.")
        if not binary and self.source_dtype not in ("|u1", "<u2"):
            dtype = np.dtype(self.source_dtype)
            if dtype.kind not in "buif" or self.parameters.scalar_range is None:
                raise ValueError("Unsupported scalar dtype in CDM artifact without explicit conversion.")
        if not binary and self.parameters.scalar_range is None and (self.source_dtype == "<u2") != (self.parameters.display_range is not None):
            raise ValueError("Artifact uint16 conversion endpoints are inconsistent.")
        if binary and self.parameters.display_range is not None:
            raise ValueError("Binary occupancy cannot have display endpoints.")
        masked = self.parameters.mode != "signal"
        if any((value is not None) != masked for value in
               (self.mask_selection, self.mask_revision, self.mask_dtype, self.mask_pixel_sha256)):
            raise ValueError("Mask identity, revision, dtype and checksum are required together.")
        if masked:
            if not isinstance(self.mask_selection, CDMSelection):
                raise ValueError("Mask selection must be explicit.")
            _text(self.mask_revision, "mask revision")
            if self.mask_dtype not in ("|b1", "|u1", "<u2"):
                raise ValueError("Unsupported mask dtype in CDM artifact.")
            if binary and self.selection != self.mask_selection:
                raise ValueError("Binary artifact selection must identify the mask.")
        if (self.png_file_sha256 is None) != (self.png_encoder is None):
            raise ValueError("PNG checksum and encoder must be recorded together.")
        if self.png_encoder is not None:
            _text(self.png_encoder, "PNG encoder")
        if not isinstance(self.warnings, (list, tuple)):
            raise ValueError("Warnings must be a sequence of messages.")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for warning in self.warnings:
            _text(warning, "warning")
        if self.mapping.evidence_id is None and not self.parameters.allow_exploratory:
            raise ValueError("Unverified mapping requires explicit exploratory output.")

    @property
    def output_label(self):
        return "exploratory" if self.mapping.evidence_id is None else "mapping_evidence_supplied"

    def to_dict(self):
        data = dict(vars(self))
        data["selection"] = self.selection.to_dict()
        data["geometry"] = self.geometry.to_dict()
        data["mask_selection"] = None if self.mask_selection is None else self.mask_selection.to_dict()
        data["mapping"] = asdict(self.mapping)
        data["parameters"] = asdict(self.parameters)
        for name, default in (("assumptions", ()), ("placement", None)):
            if data["mapping"][name] == default:
                data["mapping"].pop(name)
        if self.parameters.scalar_range is None:
            data["parameters"].pop("scalar_range")
        if self.parameters.profile == PROFILE:
            data["parameters"].pop("profile")
        if self.schema_version == 1:
            for name in ("search_canvas_shape_yx", "content_offset_yx", "anatomical_area", "alignment_space"):
                data.pop(name)
        return data

    def to_json(self):
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["selection"] = CDMSelection.from_dict(data["selection"])
        data["geometry"] = CDMGeometry.from_dict(data["geometry"])
        data["mapping"] = CDMMapping(**data["mapping"])
        data["parameters"] = CDMParameters(**data["parameters"])
        if data["mask_selection"] is not None:
            data["mask_selection"] = CDMSelection.from_dict(data["mask_selection"])
        return cls(**data)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))

    @property
    def context_sha256(self):
        """Digest of exact generation context and pixels, independent of PNG encoding."""
        data = self.to_dict()
        data.pop("png_file_sha256")
        data.pop("png_encoder")
        return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImportedCDMImage:
    """Observed file/pixel evidence; absent generation geometry stays absent."""
    source_path: str
    source_format: str
    source_file_sha256: str
    source_size: int
    imported_at: str
    shape_yx_rgb: tuple[int, int, int]
    png_file_sha256: str
    rgb_pixel_sha256: str
    manifest_json: str | None = None

    def __post_init__(self):
        _text(self.source_path, "Imported image path")
        _text(self.imported_at, "Image import time")
        if datetime.fromisoformat(self.imported_at).tzinfo is None:
            raise ValueError("Image import time must include its timezone.")
        if self.source_format not in {"PNG", "TIFF", "JPEG"}:
            raise ValueError("Unsupported Color-Depth MIP image format.")
        _integer(self.source_size, "Imported image size", 1, 8 * 1024 * 1024)
        shape = tuple(self.shape_yx_rgb)
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("Imported Color-Depth MIP must have three RGB channels.")
        for value in shape:
            _integer(value, "Imported image dimension", 1, 16 * 1024 * 1024)
        if shape[0] * shape[1] > 16 * 1024 * 1024:
            raise ValueError("Imported Color-Depth MIP exceeds the pixel limit.")
        object.__setattr__(self, "shape_yx_rgb", shape)
        for checksum in (self.source_file_sha256, self.png_file_sha256, self.rgb_pixel_sha256):
            _sha(checksum)
        if self.source_format == "PNG" and self.source_file_sha256 != self.png_file_sha256:
            raise ValueError("Imported PNG bytes must be retained unchanged.")
        if self.manifest_json is not None:
            if not isinstance(self.manifest_json, str) or len(self.manifest_json.encode("utf-8")) > 1024 * 1024:
                raise ValueError("Invalid imported Color-Depth MIP manifest.")
            try:
                manifest = json.loads(self.manifest_json)
                artifact = CDMArtifact.from_dict(manifest["artifact"])
                if (manifest["context_sha256"] != artifact.context_sha256
                        or manifest["output_shape_yx_rgb"] != list(shape)
                        or shape != (*search_profile(artifact.profile).search_canvas_yx, 3)
                        or artifact.png_file_sha256 != self.png_file_sha256
                        or artifact.rgb_pixel_sha256 != self.rgb_pixel_sha256):
                    raise ValueError("Color-Depth MIP manifest disagrees with the selected image.")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("The accompanying MADI3D Color-Depth MIP manifest is incomplete or invalid.") from exc

    @property
    def generation_artifact(self):
        return CDMArtifact.from_dict(json.loads(self.manifest_json)["artifact"]) if self.manifest_json is not None else None

    @property
    def warnings(self):
        artifact = self.generation_artifact
        warnings = list(artifact.warnings) if artifact else [
            "Imported image: template alignment and depth-color encoding are unverified. "
            "Search uses the supplied pixel canvas without resizing or registration."]
        if self.source_format == "JPEG":
            warnings.append("The imported JPEG uses lossy compression, which can affect color-depth matching.")
        if not any(self.shape_yx_rgb == (*p.search_canvas_yx, 3) for p in SEARCH_PROFILES.values()):
            warnings.append(f"This image is {self.shape_yx_rgb[1]} × {self.shape_yx_rgb[0]}; "
                            "local search requires a supported Brain or VNC canvas.")
        return tuple(warnings)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
