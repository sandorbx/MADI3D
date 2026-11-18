"""Publish a PNG and manifest together by renaming one staged artifact directory."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

from madi3d_app.volume.source_formats import SUPPORTED_VOLUME_SUFFIXES

from .cdm import CDMResult, checkpoint
from .cdm_records import CDMArtifact, canonical_json
from .search_profiles import search_profile

CDM_IMAGE_NAME = "cdm.png"
_SOURCE_SUFFIXES = (".ome.tiff", ".ome.tif", *SUPPORTED_VOLUME_SUFFIXES,
                    ".vti", ".vtk", ".vtp", ".obj", ".swc", ".ply", ".stl")


def _file_sha(path, cancel):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checkpoint(cancel)
            digest.update(chunk)
    return digest.hexdigest()


def _source_name_stem(source_name):
    """Keep meaningful dots; strip only supported input filename extensions."""
    name = str(source_name).strip()
    while name.endswith("_CDM"):
        name = name[:-4].rstrip()
    for extension in _SOURCE_SUFFIXES:
        if name.casefold().endswith(extension):
            name = name[:-len(extension)]
            break
    while name.endswith("_CDM"):
        name = name[:-4].rstrip()
    return name


def channel_name_detail(acquisition_name, channel_name):
    """Drop an acquisition prefix at a name boundary."""
    if channel_name[:len(acquisition_name)].casefold() == acquisition_name.casefold():
        tail = channel_name[len(acquisition_name):]
        if not tail or tail[0] in " _-/–—:[":
            return tail.lstrip(" _-/–—:")
    return channel_name


def automatic_cdm_label(source_name, channel_name="", *, channel_number=None):
    def elide(name, limit):
        if len(name) <= limit:
            return name
        tail = limit // 3
        return name[:limit - tail - 1].rstrip() + "…" + name[-tail:]

    source = _source_name_stem(source_name) or "Color-Depth MIP"
    channel = channel_name_detail(str(source_name).strip(), _source_name_stem(channel_name))
    channel = channel_name_detail(source, channel)
    if channel_number is not None:
        index = f"Channel {channel_number}"
        channel = f"{index} {elide(channel, 39 - len(index))}" if channel else index
    channel = elide(channel, 40)
    if channel:
        return f"{elide(source, 75 - len(channel))} {channel}_CDM"
    return elide(source, 76) + "_CDM"


@contextmanager
def stage_cdm_bundle(destination, cancel):
    """Publish the complete bundle in one sibling-directory rename."""
    checkpoint(cancel)
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=".cdm-", dir=destination.parent))
    try:
        yield staging
        checkpoint(cancel)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


def export_cdm(result: CDMResult, destination, *, cancel=None, label=None, query_id=None,
               retained_png=None) -> CDMArtifact:
    """Create a new PNG/manifest bundle; never replace an artifact.

    The sibling staging directory is private and is never a completed artifact.
    Both files are flushed/fsynced and decoded/round-trip verified before the
    single directory rename. Cancellation before that commit removes staging.
    File-system atomic rename is required; directory power-loss durability is
    platform/filesystem dependent. Consumers discover only destination bundles.
    """
    checkpoint(cancel)
    rgb = result.rgb
    if rgb.dtype != np.uint8 or rgb.shape != (*search_profile(result.artifact.profile).search_canvas_yx, 3):
        raise ValueError("Expected the full uint8 RGB template canvas.")
    pixels = rgb.tobytes()
    if hashlib.sha256(pixels).hexdigest() != result.artifact.rgb_pixel_sha256:
        raise ValueError("CDM pixels disagree with artifact checksum.")
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(rgb.shape)
    with stage_cdm_bundle(destination, cancel) as staging:
        png = staging / CDM_IMAGE_NAME
        with png.open("xb") as stream:
            if retained_png is None:
                Image.fromarray(rgb).save(stream, format="PNG", compress_level=6)
            else:
                if hashlib.sha256(retained_png).hexdigest() != result.artifact.png_file_sha256:
                    raise ValueError("Retained query PNG disagrees with its artifact checksum.")
                stream.write(retained_png)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint(cancel)
        exported = (result.artifact if retained_png is not None else
                    replace(result.artifact, png_file_sha256=_file_sha(png, cancel),
                            png_encoder=f"Pillow/{pillow_version}; PNG RGB8; compress_level=6"))
        with Image.open(png) as image:
            if image.format != "PNG" or image.mode != "RGB" or not np.array_equal(np.asarray(image), rgb):
                raise ValueError("Encoded PNG does not preserve the generated RGB pixels.")
        manifest = {"artifact": exported.to_dict(), "image": png.name,
                    "context_sha256": exported.context_sha256,
                    "output_label": exported.output_label,
                    "output_shape_yx_rgb": list(rgb.shape),
                    "pixel_checksum_encoding": "C-order interleaved RGB8",
                    "source_checksum_encoding": "C-order selected ZYX, little-endian dtype"}
        if label is not None:
            manifest["label"] = label
        if query_id is not None:
            manifest["query_id"] = query_id
        with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        loaded = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if CDMArtifact.from_dict(loaded["artifact"]) != exported:
            raise ValueError("CDM manifest failed its serialization round trip.")
    return exported


# Existing callers and saved brain workflows retain their public entry point.
export_brain_cdm = export_cdm
