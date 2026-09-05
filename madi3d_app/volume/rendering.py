"""Volume rendering presets, metadata state, and transfer-function construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import vtk


# Volume color lookup tables. The stop positions are normalized to 0..1 and
# intentionally mirror the familiar ImageJ/Fiji style of scalar-to-color maps.
# "solid" preserves MADI3D's existing single-color transfer-function behavior.
VOLUME_LUT_OPTIONS = (
    ("solid", "Solid color", None),
    ("grayscale", "Grayscale", (
        (0.00, (0.00, 0.00, 0.00)),
        (1.00, (1.00, 1.00, 1.00)),
    )),
    ("hot", "Hot", (
        (0.00, (0.00, 0.00, 0.00)),
        (0.35, (0.90, 0.00, 0.00)),
        (0.70, (1.00, 0.85, 0.00)),
        (1.00, (1.00, 1.00, 1.00)),
    )),
    ("cool", "Cool", (
        (0.00, (0.00, 1.00, 1.00)),
        (0.50, (0.35, 0.35, 1.00)),
        (1.00, (1.00, 0.00, 1.00)),
    )),
    ("hot_cold", "Hot–Cold", (
        (0.00, (0.00, 0.05, 0.65)),
        (0.25, (0.00, 0.80, 1.00)),
        (0.50, (1.00, 1.00, 1.00)),
        (0.75, (1.00, 0.80, 0.00)),
        (1.00, (0.80, 0.00, 0.00)),
    )),
    ("fire", "Fire", (
        (0.00, (0.00, 0.00, 0.00)),
        (0.25, (0.45, 0.00, 0.00)),
        (0.50, (1.00, 0.12, 0.00)),
        (0.75, (1.00, 0.80, 0.00)),
        (1.00, (1.00, 1.00, 0.85)),
    )),
    ("thermal", "Thermal", (
        (0.00, (0.00, 0.00, 0.00)),
        (0.18, (0.00, 0.00, 0.55)),
        (0.38, (0.55, 0.00, 0.75)),
        (0.58, (0.95, 0.00, 0.15)),
        (0.78, (1.00, 0.75, 0.00)),
        (1.00, (1.00, 1.00, 1.00)),
    )),
    ("rainbow", "Rainbow", (
        (0.00, (0.45, 0.00, 0.75)),
        (0.18, (0.00, 0.10, 1.00)),
        (0.36, (0.00, 0.90, 1.00)),
        (0.54, (0.00, 0.85, 0.10)),
        (0.72, (1.00, 0.95, 0.00)),
        (1.00, (1.00, 0.00, 0.00)),
    )),
    ("psychedelic", "Psychedelic", (
        (0.00, (0.10, 0.00, 0.20)),
        (0.15, (1.00, 0.00, 0.75)),
        (0.32, (0.00, 1.00, 1.00)),
        (0.50, (1.00, 1.00, 0.00)),
        (0.68, (1.00, 0.00, 0.00)),
        (0.84, (0.00, 0.20, 1.00)),
        (1.00, (1.00, 0.00, 1.00)),
    )),
    ("ice", "Ice", (
        (0.00, (0.00, 0.00, 0.05)),
        (0.35, (0.00, 0.15, 0.60)),
        (0.70, (0.00, 0.90, 1.00)),
        (1.00, (0.95, 1.00, 1.00)),
    )),
    ("spectrum", "Spectrum", (
        (0.00, (1.00, 0.00, 0.00)),
        (0.17, (1.00, 1.00, 0.00)),
        (0.34, (0.00, 1.00, 0.00)),
        (0.51, (0.00, 1.00, 1.00)),
        (0.68, (0.00, 0.00, 1.00)),
        (0.85, (1.00, 0.00, 1.00)),
        (1.00, (1.00, 0.00, 0.00)),
    )),
)
VOLUME_LUT_PRESETS = {key: stops for key, _label, stops in VOLUME_LUT_OPTIONS}
VOLUME_LUT_LABELS = {key: label for key, label, _stops in VOLUME_LUT_OPTIONS}


def normalize_volume_lut_name(value):
    key = str(value or "solid").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "grey": "grayscale",
        "gray": "grayscale",
        "greyscale": "grayscale",
        "hotcold": "hot_cold",
        "hot_cold_lut": "hot_cold",
        "colour": "solid",
        "color": "solid",
    }
    key = aliases.get(key, key)
    return key if key in VOLUME_LUT_PRESETS else "solid"


def sample_volume_lut(lut_name, position, solid_rgb=(1.0, 1.0, 1.0)):
    """Linearly sample one named volume LUT at normalized ``position``."""
    key = normalize_volume_lut_name(lut_name)
    t = max(0.0, min(1.0, float(position)))
    if key == "solid":
        try:
            return tuple(max(0.0, min(1.0, float(v))) for v in solid_rgb[:3])
        except Exception:
            return 1.0, 1.0, 1.0

    stops = VOLUME_LUT_PRESETS.get(key) or VOLUME_LUT_PRESETS["grayscale"]
    if t <= stops[0][0]:
        return tuple(float(v) for v in stops[0][1])
    if t >= stops[-1][0]:
        return tuple(float(v) for v in stops[-1][1])

    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if p0 <= t <= p1:
            span = max(1e-12, float(p1) - float(p0))
            a = (t - float(p0)) / span
            return tuple(
                float(c0[i]) + a * (float(c1[i]) - float(c0[i]))
                for i in range(3)
            )
    return tuple(float(v) for v in stops[-1][1])


def volume_lut_representative_rgb(lut_name, solid_rgb=(1.0, 1.0, 1.0)):
    key = normalize_volume_lut_name(lut_name)
    if key == "solid":
        return sample_volume_lut(key, 1.0, solid_rgb)
    return sample_volume_lut(key, 0.68, solid_rgb)


FLYBRAIN_TF_MODEL = "flybrain_fluorescence_v2"
FLYBRAIN_DATA_GAMMA_DEFAULT = 0.5
FLYBRAIN_BRIGHTNESS_DEFAULT = 1.0
FLYBRAIN_FINAL_GAMMA_DEFAULT = 4.5

# Versioned rendering-preset schema. Presets contain only fluorescence/rendering
# values; base colors and LUTs deliberately remain independent.
VOLUME_PRESET_MODEL_VERSION = 4
VOLUME_PRESET_CUSTOM = "Custom"
VOLUME_PRESET_CUSTOM_DESCRIPTION = (
    "The selected volume uses manually edited values rather than an unchanged preset."
)
VOLUME_RENDER_PRESETS = {
    "FlyBrain Standard": {
        "description": "Balanced fluorescence rendering for general anatomical volumes.",
        "lower_fraction": 0.0,
        "upper_fraction": 1.0,
        "saturation_fraction": 1.0,
        "data_gamma": 0.50,
        "brightness": 1.00,
        "global_opacity": 1.00,
        "final_gamma": 4.50,
        "threshold_softness": 0.000,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 1.00,
        "opacity_gamma_multiplier": 1.00,
        "opacity_unit_distance": 0.40,
    },
    "Weak Signal": {
        "description": "Reveals dim signal. Useful when faint structures are otherwise hard to see, but it can also expose background noise.",
        "lower_fraction": 0.0,
        "upper_fraction": 1.0,
        "saturation_fraction": 0.68,
        "data_gamma": 0.82,
        "brightness": 1.20,
        "global_opacity": 0.85,
        "final_gamma": 5.50,
        "threshold_softness": 0.002,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 1.00,
        "opacity_gamma_multiplier": 1.00,
        "opacity_unit_distance": 0.40,
    },
    "High Contrast": {
        "description": "Rejects weak background and separates medium from strong signal. Use when the image looks flat or hazy.",
        "lower_fraction": 0.040,
        "upper_fraction": 1.0,
        "saturation_fraction": 0.82,
        "data_gamma": 0.36,
        "brightness": 1.05,
        "global_opacity": 0.92,
        "final_gamma": 3.20,
        "threshold_softness": 0.003,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 1.00,
        "opacity_gamma_multiplier": 1.08,
        "opacity_unit_distance": 0.40,
    },
    "Dense Volume": {
        "description": "Reduces opacity buildup in crowded volumes so internal structures remain visible.",
        "lower_fraction": 0.010,
        "upper_fraction": 1.0,
        "saturation_fraction": 0.90,
        "data_gamma": 0.62,
        "brightness": 0.90,
        "global_opacity": 0.48,
        "final_gamma": 4.20,
        "threshold_softness": 0.004,
        "high_end_response": 0.90,
        "color_gamma_multiplier": 1.00,
        "opacity_gamma_multiplier": 1.20,
        "opacity_unit_distance": 0.65,
    },
    "Light-field Calcium Time Series": {
        "description": "For light-field calcium time series: keeps dim activity visible while limiting opacity haze during 4-D playback.",
        "lower_fraction": 0.235,
        "upper_fraction": 1.0,
        "saturation_fraction": 0.62,
        "data_gamma": 1.0,
        "brightness": 2.0,
        "global_opacity": 0.25,
        "final_gamma": 4.0,
        "threshold_softness": 0.004,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 0.47,
        "opacity_gamma_multiplier": 0.25,
        "opacity_unit_distance": 0.65,
    },
    "2-Photon Microscopy": {
        "description": "For relatively clean 2-photon stacks: moderate background rejection with strong local contrast and restrained glow.",
        "lower_fraction": 0.025,
        "upper_fraction": 1.0,
        "saturation_fraction": 0.78,
        "data_gamma": 0.58,
        "brightness": 1.00,
        "global_opacity": 0.72,
        "final_gamma": 4.30,
        "threshold_softness": 0.002,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 1.00,
        "opacity_gamma_multiplier": 0.95,
        "opacity_unit_distance": 0.45,
    },
    "Soft / Transparent": {
        "description": "Makes overlapping or reference volumes easier to see through while retaining their color context.",
        "lower_fraction": 0.0,
        "upper_fraction": 1.0,
        "saturation_fraction": 1.0,
        "data_gamma": 1.0,
        "brightness": 1.00,
        "global_opacity": 0.025,
        "final_gamma": 4.50,
        "threshold_softness": 0.004,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 0.5,
        "opacity_gamma_multiplier": 0.25,
        "opacity_unit_distance": 1.00,
    },
    "Label Field": {
        "description": "For integer label volumes and segmentation fields: removes the zero background, keeps opacity linear across label values, and reduces color response so neighboring labels retain contrast under the global glow.",
        "lower_fraction": 0.000001,
        "upper_fraction": 1.0,
        "saturation_fraction": 1.0,
        "data_gamma": 1.00,
        "brightness": 0.90,
        "global_opacity": 0.72,
        "final_gamma": 2.60,
        "threshold_softness": 0.000,
        "high_end_response": 1.00,
        "color_gamma_multiplier": 0.65,
        "opacity_gamma_multiplier": 1.00,
        "opacity_unit_distance": 0.50,
    },
}
VOLUME_PRESET_NAMES = tuple(VOLUME_RENDER_PRESETS.keys())
VOLUME_PRESET_RESOLVED_KEYS = (
    "lower_threshold", "upper_threshold", "saturation_point", "data_gamma",
    "brightness", "global_opacity", "final_gamma", "threshold_softness",
    "high_end_response", "color_gamma_multiplier", "opacity_gamma_multiplier",
    "opacity_unit_distance",
)


def _finite_float(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def normalize_volume_preset_name(value):
    raw = str(value or "").strip()
    if raw in VOLUME_RENDER_PRESETS or raw == VOLUME_PRESET_CUSTOM:
        return raw
    aliases = {
        "default": "FlyBrain Standard",
        "flybrain fluorescence": "FlyBrain Standard",
        "flybrain standard fluorescence": "FlyBrain Standard",
        "weak": "Weak Signal",
        "contrast": "High Contrast",
        "dense": "Dense Volume",
        "lightfield": "Light-field Calcium Time Series",
        "light-field": "Light-field Calcium Time Series",
        "light field calcium": "Light-field Calcium Time Series",
        "calcium time series": "Light-field Calcium Time Series",
        "2p": "2-Photon Microscopy",
        "two photon": "2-Photon Microscopy",
        "2-photon": "2-Photon Microscopy",
        "soft": "Soft / Transparent",
        "transparent": "Soft / Transparent",
        "binary": "Label Field",
        "mask": "Label Field",
        "binary / mask": "Label Field",
        "label": "Label Field",
        "labels": "Label Field",
        "label field": "Label Field",
        "custom (mixed)": VOLUME_PRESET_CUSTOM,
    }
    return aliases.get(raw.lower(), VOLUME_PRESET_CUSTOM)


def resolve_volume_preset(preset_name, data_min, data_max):
    """Resolve normalized preset thresholds to one volume's scalar range."""
    preset_name = normalize_volume_preset_name(preset_name)
    spec = VOLUME_RENDER_PRESETS.get(preset_name)
    if spec is None:
        return None
    lo = float(data_min)
    hi = float(data_max)
    if hi < lo:
        lo, hi = hi, lo
    span = max(1e-12, hi - lo)
    resolved = {
        "lower_threshold": lo + span * float(spec["lower_fraction"]),
        "upper_threshold": lo + span * float(spec["upper_fraction"]),
        "saturation_point": lo + span * float(spec["saturation_fraction"]),
    }
    for key in VOLUME_PRESET_RESOLVED_KEYS:
        if key not in resolved and key in spec:
            resolved[key] = float(spec[key])
    resolved["peak"] = resolved["saturation_point"]
    resolved["preset_name"] = preset_name
    resolved["preset_version"] = VOLUME_PRESET_MODEL_VERSION
    return resolved


def volume_preset_values_match(metadata, preset_name, final_gamma=None):
    """Return True when resolved metadata still matches a named preset."""
    md = dict(metadata or {})
    lo = _finite_float(md.get("data_min"), 0.0)
    hi = _finite_float(md.get("data_max"), lo + 1.0)
    resolved = resolve_volume_preset(preset_name, lo, hi)
    if resolved is None:
        return False
    span = max(1e-12, abs(float(hi) - float(lo)))
    scalar_keys = {"lower_threshold", "upper_threshold", "saturation_point"}
    for key in VOLUME_PRESET_RESOLVED_KEYS:
        expected = resolved.get(key)
        if key == "final_gamma" and final_gamma is not None:
            actual = _finite_float(final_gamma, None)
        else:
            actual = _finite_float(md.get(key), None)
        if actual is None or expected is None:
            return False
        tolerance = max(1e-7, span * 1e-6) if key in scalar_keys else 1e-6
        if abs(float(actual) - float(expected)) > tolerance:
            return False
    return True


def migrate_volume_transfer_metadata(metadata, data_min=None, data_max=None):
    """Convert current or legacy volume metadata to the current transfer model.

    Legacy CSV values are retained only as approximate resolved settings. They
    are marked Custom because they did not originate from the versioned preset
    schema. Obsolete transfer-function fields are then discarded.
    """
    md = dict(metadata or {})
    legacy_keys = {
        "color_gamma", "opacity_gamma", "base_alpha",
        "shift_opacity", "shift_color",
    }
    legacy_tf = (
        str(md.get("tf_model", "")).strip() not in (FLYBRAIN_TF_MODEL, "flybrain_fluorescence_v1")
        or bool(legacy_keys.intersection(md))
        or ("data_gamma" not in md and "peak" in md)
    )

    lo = _finite_float(data_min, _finite_float(md.get("data_min"), 0.0))
    hi = _finite_float(data_max, _finite_float(md.get("data_max"), lo + 1.0))
    if hi is None or lo is None:
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0

    lower = _finite_float(md.get("lower_threshold"), lo)
    upper = _finite_float(md.get("upper_threshold"), hi)
    saturation = _finite_float(md.get("saturation_point", md.get("peak")), hi)

    # Estimate one threshold range from legacy color/opacity offsets.
    if "data_gamma" not in md:
        shift_color = _finite_float(md.get("shift_color"), 0.0) or 0.0
        shift_opacity = _finite_float(md.get("shift_opacity"), 0.0) or 0.0
        lower += 0.5 * (shift_color + shift_opacity)
        upper += shift_opacity

    gamma = _finite_float(md.get("data_gamma"), None)
    if gamma is None:
        legacy_gammas = [
            value for value in (
                _finite_float(md.get("color_gamma"), None),
                _finite_float(md.get("opacity_gamma"), None),
            )
            if value is not None and value > 0.0
        ]
        gamma = (sum(legacy_gammas) / len(legacy_gammas)
                 if legacy_gammas else FLYBRAIN_DATA_GAMMA_DEFAULT)

    lower = max(lo, min(hi, lower))
    upper = max(lo, min(hi, upper))
    if upper < lower:
        lower, upper = upper, lower
    saturation = max(lo, min(hi, saturation))

    md.update({
        "tf_model": FLYBRAIN_TF_MODEL,
        "data_min": lo,
        "data_max": hi,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "saturation_point": saturation,
        "peak": saturation,
        "data_gamma": max(0.05, min(10.0, float(gamma))),
        "brightness": max(0.0, min(12.0, float(
            _finite_float(md.get("brightness"), FLYBRAIN_BRIGHTNESS_DEFAULT)
        ))),
        "global_opacity": max(0.0, min(1.0, float(
            _finite_float(md.get("global_opacity"), 1.0)
        ))),
        "final_gamma": max(0.1, min(10.0, float(
            _finite_float(md.get("final_gamma"), FLYBRAIN_FINAL_GAMMA_DEFAULT)
        ))),
        # Fractions are relative to the volume's scalar range.
        "threshold_softness": max(0.0, min(0.25, float(
            _finite_float(md.get("threshold_softness"), 0.0)
        ))),
        "high_end_response": max(0.0, min(1.0, float(
            _finite_float(md.get("high_end_response"), 1.0)
        ))),
        "color_gamma_multiplier": max(0.1, min(4.0, float(
            _finite_float(md.get("color_gamma_multiplier"), 1.0)
        ))),
        "opacity_gamma_multiplier": max(0.1, min(4.0, float(
            _finite_float(md.get("opacity_gamma_multiplier"), 1.0)
        ))),
        "opacity_unit_distance": max(0.001, min(5.0, float(
            _finite_float(md.get("opacity_unit_distance"), 0.4)
        ))),
    })

    stored_name = normalize_volume_preset_name(
        md.get("preset_name", md.get("render_preset"))
    )
    stored_version = int(_finite_float(
        md.get("preset_version", md.get("render_preset_version")), 0
    ) or 0)
    if legacy_tf or stored_version not in (0, VOLUME_PRESET_MODEL_VERSION):
        active_name = VOLUME_PRESET_CUSTOM
    elif stored_name in VOLUME_RENDER_PRESETS and volume_preset_values_match(
        md, stored_name, md.get("final_gamma")
    ):
        active_name = stored_name
    elif stored_version == 0 and volume_preset_values_match(
        md, "FlyBrain Standard", md.get("final_gamma")
    ):
        active_name = "FlyBrain Standard"
    else:
        active_name = VOLUME_PRESET_CUSTOM
    md["preset_name"] = active_name
    md["preset_version"] = VOLUME_PRESET_MODEL_VERSION
    md.pop("render_preset", None)
    md.pop("render_preset_version", None)

    # Gradient-opacity controls were removed because they can select an
    # unstable vtkMultiVolume shader path. Old CSV values are accepted but ignored.
    legacy_keys.update({"gradient_threshold", "gradient_falloff"})
    for obsolete in legacy_keys:
        md.pop(obsolete, None)
    return md


VOLUME_SCALAR_DOMAIN_METADATA_KEYS = (
    "data_min",
    "data_max",
)

VOLUME_SCALAR_COORDINATE_METADATA_KEYS = (
    "lower_threshold",
    "upper_threshold",
    "saturation_point",
    "peak",
    "iso_value",
)

VOLUME_SCALAR_DEPENDENT_METADATA_KEYS = (
    *VOLUME_SCALAR_DOMAIN_METADATA_KEYS,
    *VOLUME_SCALAR_COORDINATE_METADATA_KEYS,
)


def resolve_volume_scalar_range(container):
    """Return a resident volume's scalar domain without redundant VTK scans.

    Decoding records the domain in rendering metadata and on the runtime
    container.  Query the image only for unusual containers that lack both
    valid cached representations.
    """
    metadata = getattr(container, "metadata", {}) or {}
    cached_ranges = (
        (metadata.get("data_min"), metadata.get("data_max")),
        getattr(container, "original_scalar_range", None),
    )
    for values in cached_ranges:
        if values is None:
            continue
        try:
            lo, hi = float(values[0]), float(values[1])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if math.isfinite(lo) and math.isfinite(hi) and hi >= lo:
            return lo, hi

    image = getattr(container, "image", None)
    if image is None:
        raise ValueError("Volume has no cached scalar range or resident image.")
    lo, hi = image.GetScalarRange()
    return float(lo), float(hi)


def rebase_saved_volume_scalar_metadata(metadata, actual_min, actual_max):
    """Rebase saved scalar-valued settings onto the range read from the file.

    ``data_min``/``data_max`` describe the source data domain; they are not a
    user rendering preference.  CSV values can therefore be used only as the
    old coordinate system for settings such as cutoffs, never as authority over
    the range measured from the volume being loaded.

    This also repairs projects produced by older builds that explicitly wrote
    an unloaded volume as 0..1: values such as 0/1 cutoffs are mapped back to
    the real data range when the volume is finally loaded.
    """
    md = dict(metadata or {})
    actual_lo = _finite_float(actual_min, 0.0)
    actual_hi = _finite_float(actual_max, actual_lo + 1.0)
    if actual_hi < actual_lo:
        actual_lo, actual_hi = actual_hi, actual_lo
    if abs(actual_hi - actual_lo) < 1e-12:
        actual_hi = actual_lo + 1.0

    saved_lo = _finite_float(md.get("data_min"), None)
    saved_hi = _finite_float(md.get("data_max"), None)
    if saved_lo is not None and saved_hi is not None and saved_hi < saved_lo:
        saved_lo, saved_hi = saved_hi, saved_lo

    if (
        saved_lo is not None and saved_hi is not None
        and abs(saved_hi - saved_lo) > 1e-12
        and (
            abs(saved_lo - actual_lo) > max(1e-9, abs(actual_hi - actual_lo) * 1e-9)
            or abs(saved_hi - actual_hi) > max(1e-9, abs(actual_hi - actual_lo) * 1e-9)
        )
    ):
        saved_span = saved_hi - saved_lo
        actual_span = actual_hi - actual_lo
        tolerance = max(1e-9, abs(saved_span) * 1e-6)
        for key in VOLUME_SCALAR_COORDINATE_METADATA_KEYS:
            value = _finite_float(md.get(key), None)
            if value is None:
                continue
            # These controls are range-bounded in the UI. Rebase only values
            # that plausibly lived in the saved scalar coordinate system.
            if saved_lo - tolerance <= value <= saved_hi + tolerance:
                fraction = (value - saved_lo) / saved_span
                fraction = max(0.0, min(1.0, fraction))
                md[key] = actual_lo + fraction * actual_span

    # The file-derived range always wins.
    md["data_min"] = actual_lo
    md["data_max"] = actual_hi
    return md


VOLUME_RENDERING_METADATA_KEYS = (
    "data_min",
    "data_max",
    "lower_threshold",
    "upper_threshold",
    "tf_model",
    "preset_name",
    "preset_version",
    "saturation_point",
    "peak",
    "data_gamma",
    "brightness",
    "threshold_softness",
    "high_end_response",
    "color_gamma_multiplier",
    "opacity_gamma_multiplier",
    "final_gamma",
    "opacity_unit_distance",
    "color",
    "color_lut",
    "global_opacity",
    "iso_value",
    "representation",
    "time_index",
)


def default_volume_rendering_metadata(data_min, data_max):
    """Return the canonical rendering state for a newly loaded volume."""
    lo = float(data_min)
    hi = float(data_max)
    if hi < lo:
        lo, hi = hi, lo
    return {
        "data_min": lo,
        "data_max": hi,
        "lower_threshold": lo,
        "upper_threshold": hi,
        "tf_model": FLYBRAIN_TF_MODEL,
        "preset_name": "FlyBrain Standard",
        "preset_version": VOLUME_PRESET_MODEL_VERSION,
        "saturation_point": hi,
        "peak": hi,
        "data_gamma": FLYBRAIN_DATA_GAMMA_DEFAULT,
        "brightness": FLYBRAIN_BRIGHTNESS_DEFAULT,
        "threshold_softness": 0.0,
        "high_end_response": 1.0,
        "color_gamma_multiplier": 1.0,
        "opacity_gamma_multiplier": 1.0,
        "final_gamma": FLYBRAIN_FINAL_GAMMA_DEFAULT,
        "opacity_unit_distance": 0.4,
        "color": (1.0, 1.0, 1.0),
        "color_lut": "solid",
        "global_opacity": 1.0,
        "iso_value": 0.2 * (lo + hi),
        "representation": "volume",
        "time_index": 0,
    }


def serialize_volume_rendering_metadata(metadata):
    """Return a detached, stable rendering-only metadata mapping."""
    md = migrate_volume_transfer_metadata(metadata)
    serialized = {
        key: md[key]
        for key in VOLUME_RENDERING_METADATA_KEYS
        if key in md
    }
    if "color" in serialized:
        serialized["color"] = tuple(float(value) for value in serialized["color"][:3])
    serialized["color_lut"] = normalize_volume_lut_name(
        serialized.get("color_lut", "solid")
    )
    representation = str(serialized.get("representation", "volume")).strip().lower()
    serialized["representation"] = (
        representation if representation in {"volume", "surface"} else "volume"
    )
    return serialized


def deserialize_volume_rendering_metadata(metadata, data_min=None, data_max=None):
    """Normalize persisted rendering state without accepting unrelated metadata."""
    source = dict(metadata or {})
    if data_min is not None and data_max is not None:
        source = rebase_saved_volume_scalar_metadata(source, data_min, data_max)
    normalized = migrate_volume_transfer_metadata(source, data_min, data_max)
    defaults = default_volume_rendering_metadata(
        normalized["data_min"], normalized["data_max"]
    )
    defaults.update(serialize_volume_rendering_metadata(normalized))
    return defaults


@dataclass
class VolumeRenderingState:
    """Owned rendering metadata with an explicit persistence boundary."""

    metadata: dict[str, Any]

    @classmethod
    def defaults(cls, data_min, data_max):
        return cls(default_volume_rendering_metadata(data_min, data_max))

    @classmethod
    def from_metadata(cls, metadata, data_min=None, data_max=None):
        return cls(
            deserialize_volume_rendering_metadata(metadata, data_min, data_max)
        )

    def to_metadata(self):
        return serialize_volume_rendering_metadata(self.metadata)

    def apply_preset(self, preset_name):
        values = resolve_volume_preset(
            preset_name,
            self.metadata["data_min"],
            self.metadata["data_max"],
        )
        if values is None:
            raise ValueError(f"Unknown volume rendering preset: {preset_name!r}")
        self.metadata.update(values)
        return dict(values)


class VolumeTransferFunctionModel:
    """Construct volume transfer functions from rendering metadata."""

    def __init__(self, max_lookup_table_size=32768):
        self.max_lookup_table_size = int(max_lookup_table_size)

    @staticmethod
    def clamp_transfer_function_size(transfer_function, render_window, desired_size):
        render_window.Render()
        ogl_window = vtk.vtkOpenGLRenderWindow.SafeDownCast(render_window)
        max_width = vtk.vtkOpenGLVolumeLookupTable.GetMaximumSupportedTextureWidth(
            ogl_window, desired_size
        )
        transfer_function.SetNumberOfTableValues(min(desired_size, max_width))

    @staticmethod
    def scalar_response_array(
        values,
        data_min,
        data_max,
        saturation_point,
        data_gamma,
        high_end_response=1.0,
    ):
        """Generalized VVD/Janelia response with an optional post-peak tail."""
        lo, hi = float(data_min), float(data_max)
        span = max(1e-12, hi - lo)
        scalar_norm = np.clip((np.asarray(values, dtype=float) - lo) / span, 0.0, 1.0)
        peak = np.clip((float(saturation_point) - lo) / span, 0.0, 1.0)
        gamma = max(1e-6, float(data_gamma))
        tail_response = np.clip(float(high_end_response), 0.0, 1.0)

        if peak <= 1e-12:
            rise = np.ones_like(scalar_norm)
        else:
            rise = np.clip(scalar_norm / peak, 0.0, 1.0) ** (1.0 / gamma)

        if peak >= 1.0 - 1e-12:
            return rise
        tail_position = np.clip(
            (scalar_norm - peak) / max(1e-12, 1.0 - peak), 0.0, 1.0
        )
        tail = 1.0 + tail_position * (tail_response - 1.0)
        return np.where(scalar_norm <= peak, rise, tail)

    @staticmethod
    def _smoothstep_array(edge0, edge1, values):
        if edge1 <= edge0:
            return (np.asarray(values, dtype=float) >= edge1).astype(float)
        position = np.clip(
            (np.asarray(values, dtype=float) - edge0) / (edge1 - edge0),
            0.0,
            1.0,
        )
        return position * position * (3.0 - 2.0 * position)

    @classmethod
    def threshold_gate_array(
        cls,
        values,
        data_min,
        data_max,
        lower_threshold,
        upper_threshold,
        threshold_softness=0.0,
    ):
        lo, hi = float(data_min), float(data_max)
        lower = max(lo, min(hi, float(lower_threshold)))
        upper = max(lo, min(hi, float(upper_threshold)))
        if upper < lower:
            lower, upper = upper, lower
        values = np.asarray(values, dtype=float)
        span = max(1e-12, hi - lo)
        softness = max(0.0, min(0.25, float(threshold_softness))) * span
        epsilon = max(span * 1e-9, np.finfo(float).eps)
        if softness <= epsilon:
            return ((values >= lower) & (values <= upper)).astype(float)

        low_gate = np.ones_like(values)
        high_gate = np.ones_like(values)
        if lower > lo + epsilon:
            low_gate = cls._smoothstep_array(
                lower - softness, lower + softness, values
            )
        if upper < hi - epsilon:
            high_gate = 1.0 - cls._smoothstep_array(
                upper - softness, upper + softness, values
            )
        return np.clip(low_gate * high_gate, 0.0, 1.0)

    @staticmethod
    def sample_positions(
        data_min,
        data_max,
        lower_threshold,
        upper_threshold,
        saturation_point,
        count,
        threshold_softness=0.0,
    ):
        lo, hi = float(data_min), float(data_max)
        span = max(1e-12, hi - lo)
        epsilon = max(
            span * 1e-6, np.finfo(float).eps * max(1.0, abs(lo), abs(hi))
        )
        softness = max(0.0, min(0.25, float(threshold_softness))) * span
        points = list(np.linspace(lo, hi, num=max(32, int(count))))
        for value in (lower_threshold, upper_threshold, saturation_point):
            points.append(max(lo, min(hi, float(value))))
        for threshold in (lower_threshold, upper_threshold):
            threshold = max(lo, min(hi, float(threshold)))
            for delta in (-softness, -epsilon, epsilon, softness):
                points.append(max(lo, min(hi, threshold + delta)))
        return np.array(sorted(set(float(value) for value in points)), dtype=float)

    def build_color_transfer_function(
        self,
        lower_threshold,
        upper_threshold,
        saturation_point,
        data_gamma,
        color_rgb,
        data_min,
        data_max,
        brightness=1.0,
        target=None,
        lut_name="solid",
        threshold_softness=0.0,
        high_end_response=1.0,
        gamma_multiplier=1.0,
    ):
        lower = max(float(data_min), min(float(data_max), float(lower_threshold)))
        upper = max(float(data_min), min(float(data_max), float(upper_threshold)))
        if upper < lower:
            lower, upper = upper, lower
        saturation = max(
            float(data_min), min(float(data_max), float(saturation_point))
        )
        lut_name = normalize_volume_lut_name(lut_name)
        base_rgb = tuple(max(0.0, min(1.0, float(value))) for value in color_rgb[:3])
        brightness = max(0.0, min(12.0, float(brightness)))
        effective_gamma = max(0.01, float(data_gamma) * float(gamma_multiplier))

        values = self.sample_positions(
            data_min,
            data_max,
            lower,
            upper,
            saturation,
            256,
            threshold_softness,
        )
        response = np.nan_to_num(
            self.scalar_response_array(
                values,
                data_min,
                data_max,
                saturation,
                effective_gamma,
                high_end_response,
            ),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        gate = np.nan_to_num(
            self.threshold_gate_array(
                values,
                data_min,
                data_max,
                lower,
                upper,
                threshold_softness,
            ),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        transfer_function = target if target is not None else vtk.vtkColorTransferFunction()
        transfer_function.RemoveAllPoints()
        for scalar, response_value, visibility in zip(values, response, gate):
            weight = float(response_value) * float(visibility) * brightness
            source_rgb = (
                base_rgb
                if lut_name == "solid"
                else sample_volume_lut(lut_name, float(response_value), base_rgb)
            )
            rgb = tuple(
                min(1.0, max(0.0, float(channel) * weight))
                for channel in source_rgb
            )
            transfer_function.AddRGBPoint(float(scalar), *rgb)
        transfer_function.Modified()
        return transfer_function

    def build_opacity_transfer_function(
        self,
        lower_threshold,
        upper_threshold,
        saturation_point,
        data_gamma,
        data_min,
        data_max,
        global_opacity=1.0,
        target=None,
        threshold_softness=0.0,
        high_end_response=1.0,
        gamma_multiplier=1.0,
    ):
        lower = max(float(data_min), min(float(data_max), float(lower_threshold)))
        upper = max(float(data_min), min(float(data_max), float(upper_threshold)))
        if upper < lower:
            lower, upper = upper, lower
        saturation = max(
            float(data_min), min(float(data_max), float(saturation_point))
        )
        opacity_scale = max(0.0, min(1.0, float(global_opacity)))
        effective_gamma = max(0.01, float(data_gamma) * float(gamma_multiplier))

        values = self.sample_positions(
            data_min,
            data_max,
            lower,
            upper,
            saturation,
            512,
            threshold_softness,
        )
        response = np.nan_to_num(
            self.scalar_response_array(
                values,
                data_min,
                data_max,
                saturation,
                effective_gamma,
                high_end_response,
            ),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        gate = np.nan_to_num(
            self.threshold_gate_array(
                values,
                data_min,
                data_max,
                lower,
                upper,
                threshold_softness,
            ),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        opacity = np.clip(response * gate * opacity_scale, 0.0, 1.0)

        if opacity_scale > 0.0 and upper > lower and float(np.max(opacity)) <= 1e-12:
            response = self.scalar_response_array(
                values,
                data_min,
                data_max,
                saturation,
                max(0.01, float(data_gamma)),
                1.0,
            )
            gate = self.threshold_gate_array(
                values, data_min, data_max, lower, upper, 0.0
            )
            opacity = np.clip(response * gate * opacity_scale, 0.0, 1.0)

        transfer_function = target if target is not None else vtk.vtkPiecewiseFunction()
        transfer_function.RemoveAllPoints()
        for scalar, alpha in zip(values, opacity):
            transfer_function.AddPoint(float(scalar), float(alpha))
        transfer_function.Modified()
        return transfer_function

    def rebuild(
        self,
        metadata,
        transfer_functions,
        volume_property,
        rebuild_color=True,
        rebuild_opacity=True,
    ):
        """Normalize state and rebuild the requested transfer-function channels."""
        md = migrate_volume_transfer_metadata(
            metadata,
            metadata.get("data_min"),
            metadata.get("data_max"),
        )
        lo, hi = float(md["data_min"]), float(md["data_max"])
        saturation = float(md.get("saturation_point", md.get("peak", hi)))
        md["saturation_point"] = saturation
        md["peak"] = saturation
        common = {
            "lower_threshold": float(md["lower_threshold"]),
            "upper_threshold": float(md["upper_threshold"]),
            "saturation_point": saturation,
            "data_gamma": float(md["data_gamma"]),
            "data_min": lo,
            "data_max": hi,
            "threshold_softness": float(md.get("threshold_softness", 0.0)),
            "high_end_response": float(md.get("high_end_response", 1.0)),
        }

        if rebuild_color:
            self.build_color_transfer_function(
                color_rgb=md["color"],
                brightness=float(md.get("brightness", FLYBRAIN_BRIGHTNESS_DEFAULT)),
                gamma_multiplier=float(md.get("color_gamma_multiplier", 1.0)),
                target=transfer_functions["color"],
                lut_name=md.get("color_lut", "solid"),
                **common,
            )
        if rebuild_opacity:
            self.build_opacity_transfer_function(
                global_opacity=float(md["global_opacity"]),
                gamma_multiplier=float(md.get("opacity_gamma_multiplier", 1.0)),
                target=transfer_functions["opacity"],
                **common,
            )
            self.update_opacity_unit_distance(md, volume_property)
        return md

    @staticmethod
    def update_opacity_unit_distance(metadata, volume_property):
        volume_property.SetScalarOpacityUnitDistance(
            float(metadata["opacity_unit_distance"])
        )

    def scalar_response(self, metadata, scalar_value):
        values = self.scalar_response_array(
            [scalar_value],
            metadata.get("data_min", 0.0),
            metadata.get("data_max", 1.0),
            metadata.get("saturation_point", metadata.get("peak", 1.0)),
            float(metadata.get("data_gamma", FLYBRAIN_DATA_GAMMA_DEFAULT))
            * float(metadata.get("color_gamma_multiplier", 1.0)),
            metadata.get("high_end_response", 1.0),
        )
        return float(values[0])

    def surface_display_color(self, metadata):
        lut_name = normalize_volume_lut_name(metadata.get("color_lut", "solid"))
        base_rgb = metadata.get("color", (1.0, 1.0, 1.0))
        brightness = max(
            0.0,
            min(12.0, float(metadata.get("brightness", FLYBRAIN_BRIGHTNESS_DEFAULT))),
        )
        if lut_name == "solid":
            rgb = sample_volume_lut("solid", 1.0, base_rgb)
        else:
            rgb = sample_volume_lut(
                lut_name,
                self.scalar_response(
                    metadata,
                    metadata.get("iso_value", metadata.get("saturation_point", 0.0)),
                ),
                base_rgb,
            )
        return tuple(
            min(1.0, max(0.0, float(channel) * brightness))
            for channel in rgb
        )
