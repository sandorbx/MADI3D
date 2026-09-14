"""Publish a PNG and manifest together by renaming one staged artifact directory."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from .cdm import CDMResult, checkpoint
from .cdm_records import CDMArtifact, canonical_json
from .search_profiles import search_profile


def _file_sha(path, cancel):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checkpoint(cancel)
            digest.update(chunk)
    return digest.hexdigest()


def cdm_image_name(label=None):
    """Return a portable descriptive filename, independent of scientific IDs."""
    if not label:
        return "cdm.png"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', '-', str(label)).strip(" .")
    stem = stem.encode("utf-8")[:96].decode("utf-8", errors="ignore").rstrip(" .") or "Color-Depth MIP"
    return stem + "-cdm.png"


def export_cdm(result: CDMResult, destination, *, cancel=None, label=None) -> CDMArtifact:
    """Create a new PNG/manifest bundle; never replace an artifact.

    The sibling staging directory is private and is never a completed artifact.
    Both files are flushed/fsynced and decoded/round-trip verified before the
    single directory rename. Cancellation before that commit removes staging.
    File-system atomic rename is required; directory power-loss durability is
    platform/filesystem dependent. Consumers discover only destination bundles.
    """
    checkpoint(cancel)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise ValueError("Artifact parent directory must already exist.")
    rgb = result.rgb
    if rgb.dtype != np.uint8 or rgb.shape != (*search_profile(result.artifact.profile).search_canvas_yx, 3):
        raise ValueError("Expected the full uint8 RGB template canvas.")
    pixels = rgb.tobytes()
    if hashlib.sha256(pixels).hexdigest() != result.artifact.rgb_pixel_sha256:
        raise ValueError("CDM pixels disagree with artifact checksum.")
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rgb.shape)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.pending-", dir=destination.parent))
    try:
        image_name = cdm_image_name(label)
        png = staging / image_name
        with png.open("xb") as stream:
            Image.fromarray(rgb).save(stream, format="PNG", compress_level=6)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint(cancel)
        exported = replace(result.artifact, png_file_sha256=_file_sha(png, cancel),
                           png_encoder=f"Pillow/{pillow_version}; PNG RGB8; compress_level=6")
        with Image.open(png) as image:
            if image.mode != "RGB" or not np.array_equal(np.asarray(image), rgb):
                raise ValueError("Encoded PNG does not preserve the generated RGB pixels.")
        manifest = {"artifact": exported.to_dict(), "image": image_name,
                    "context_sha256": exported.context_sha256,
                    "output_label": exported.output_label,
                    "output_shape_yx_rgb": list(rgb.shape),
                    "pixel_checksum_encoding": "C-order interleaved RGB8",
                    "source_checksum_encoding": "C-order selected ZYX, little-endian dtype"}
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        loaded = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if CDMArtifact.from_dict(loaded["artifact"]) != exported:
            raise ValueError("CDM manifest failed its serialization round trip.")
        checkpoint(cancel)
        if destination.exists():
            raise FileExistsError(destination)
        # os.rename does not replace an existing nonempty artifact directory.
        os.rename(staging, destination)
        return exported
    finally:
        if staging.exists():
            shutil.rmtree(staging)


# Existing callers and saved brain workflows retain their public entry point.
export_brain_cdm = export_cdm
