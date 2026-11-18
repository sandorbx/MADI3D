"""Opt-in NB-02 oracle runner; never imported by MADI3D.

Downloads hash-pinned upstream tools into --work-dir, then executes the original
ImageJ plugins. Requires numpy, Pillow, tifffile and Java 8 (a JRE is sufficient).
On Linux use xvfb-run: upstream GenericDialog requires AWT even in batch mode.
Offline tests import only fixture staging and parity/identity checks.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

import numpy as np
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/neuronbridge_colordepth"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_manifest_identity(manifest):
    """Refuse changed reference glue/recipes before executing the oracle."""
    files = {
        "cases_sha256": FIXTURES / "cases.json",
        "reference_lock_sha256": FIXTURES / "reference-lock.json",
        "launcher_sha256": Path(__file__).with_name("RunReference.java"),
        "runner_sha256": Path(__file__),
    }
    for key, path in files.items():
        data = path.read_bytes()
        if key in ("launcher_sha256", "runner_sha256"):
            data = data.replace(b"\r\n", b"\n")
        if sha(data) != manifest[key]:
            raise AssertionError("Reference identity changed: " + path.name)


def read_input(case):
    """Expand fixture data only; no depth encoding or RGB projection here."""
    dtype = np.dtype(case["dtype"])
    shape = tuple(case["shape_zyx"])
    if case["kind"] == "rgb_projection":
        shape += (3,)
    if "data_file" in case:
        raw = gzip.decompress((FIXTURES / case["data_file"]).read_bytes())
        data = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    else:
        data = np.zeros(shape, dtype=dtype)
        for z, y, x, value in case.get("voxels_zyxv", []):
            data[z, y, x] = value
    return data


def place_on_canvas(data, case):
    """Fixture staging for explicitly indexed crops, never an image resampler."""
    if "offset_xyz" not in case:
        return data
    steps = case.get("index_step_xyz", [1, 1, 1])
    if any(step not in (-1, 1) for step in steps):
        raise ValueError("Fixture index steps must be +1 or -1")
    axes = [start + step * np.arange(size) for start, step, size in
            zip(case["offset_xyz"][::-1], steps[::-1], data.shape)]
    if any(a.min() < 0 or a.max() >= size for a, size in
           zip(axes, case["full_shape_zyx"])):
        raise ValueError("Fixture crop lies outside the declared canvas")
    padded = np.zeros(case["full_shape_zyx"], dtype=data.dtype)
    padded[np.ix_(*axes)] = data
    return padded


def download(entry, directory):
    path = directory / entry["name"]
    if not path.exists():
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            content = response.read()
        if sha(content) != entry["sha256"]:
            raise ValueError("Download hash mismatch: " + entry["name"])
        path.write_bytes(content)
    if sha(path.read_bytes()) != entry["sha256"]:
        raise ValueError("Cached reference hash mismatch: " + str(path))
    return path


def quote(path):
    return str(path).replace("\\", "/").replace('"', '\\"')


def macro_for(case, directory, tools_dir):
    lines = ["setBatchMode(true);", 'setOption("ScaleConversions", true);',
             f'open("{quote(directory / "input.tif")}");',
             'rename("Original_Stack.tif");']
    if case["kind"] != "rgb_projection":
        if case["dtype"] == "<u2":
            lo, hi = case["display_range"]
            lines += [f"setMinAndMax({lo}, {hi});", 'run("8-bit");']
        if "mask_zyx" in case:
            lines += [f'open("{quote(directory / "mask.tif")}");',
                      'rename("mask.tif");',
                      'imageCalculator("Multiply stack", "Original_Stack.tif", "mask.tif");',
                      'selectWindow("mask.tif"); close();',
                      'selectWindow("Original_Stack.tif");']
        if case.get("threshold", 0):
            # Explicit NB-02 input policy, performed by ImageJ. Retain amplitudes.
            threshold = case["threshold"]
            lines += ["for(s=1;s<=nSlices;s++) {setSlice(s); changeValues(0," +
                      str(threshold) + ",0);}"]
        if case.get("upstream_reverse", False):
            lines += ['run("Reverse");', 'run("Flip Horizontally", "stack");']
        lines += [f'saveAs("Tiff", "{quote(directory / "prepared.tif")}");',
                  'rename("Original_Stack.tif");',
                  "N=nSlices;",
                  'newImage("lut_table.tif", "8-bit black", N, 1, 1);',
                  "for(z=0;z<N;z++) setPixel(z,0,round(255*(z/N)));",
                  f'open("{quote(tools_dir / "PsychedelicRainBow2.lut")}");',
                  'run("RGB Color");',
                  f'saveAs("PNG", "{quote(directory / "depth_lut.png")}");',
                  'rename("lut_table.tif");',
                  'selectWindow("Original_Stack.tif");',
                  'run("Z Code Stack HO", "data=Original_Stack.tif 1px=lut_table.tif");',
                  'selectWindow("Depth_color_RGB.tif");']
    start, end = case.get("projection_slices_1based", [1, case["shape_zyx"][0]])
    lines += [f'run("MIP right color", "start={start} end={end}");',
              f'saveAs("PNG", "{quote(directory / "output.png")}");',
              'run("Close All");']
    return "\n".join(lines) + "\n"


def assert_parity(actual, expected):
    """NB-03A gate, fixed before native implementation: zero RGB error."""
    if actual.dtype != np.uint8 or actual.shape != expected.shape:
        raise AssertionError("Expected uint8 H x W x 3 with exact canvas dimensions")
    ref_fg = np.any(expected != 0, axis=-1)
    got_fg = np.any(actual != 0, axis=-1)
    union = ref_fg | got_fg
    false_positive = int(np.count_nonzero(got_fg & ~ref_fg))
    false_negative = int(np.count_nonzero(ref_fg & ~got_fg))
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    mismatch = int(np.count_nonzero(np.any(error != 0, axis=-1) & union))
    max_error = int(error[union].max()) if union.any() else 0
    if false_positive or false_negative or mismatch:
        raise AssertionError(
            f"foreground FP={false_positive}, FN={false_negative}, "
            f"RGB mismatches={mismatch}, max channel error={max_error}")
    return {"foreground_pixels": int(ref_fg.sum()), "foreground_mismatches": mismatch,
            "false_positive": false_positive, "false_negative": false_negative,
            "max_channel_error": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--record", action="store_true",
                        help="Explicitly replace committed expected outputs/manifest")
    parser.add_argument("--case", action="append", help="Run only named cases (verify only)")
    args = parser.parse_args()
    if args.record and args.case:
        parser.error("--record must regenerate the entire corpus")
    cases = json.loads((FIXTURES / "cases.json").read_text())
    unknown = set(args.case or []) - {case["name"] for case in cases}
    if unknown:
        parser.error("Unknown cases: " + ", ".join(sorted(unknown)))
    manifest_path = FIXTURES / "manifest.json"
    if not args.record:
        manifest = json.loads(manifest_path.read_text())
        verify_manifest_identity(manifest)
    work = args.work_dir.resolve()
    if work == ROOT or ROOT in work.parents:
        parser.error("Keep downloaded tools and intermediates outside the repository")
    tool_dir = work / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    pins = json.loads((FIXTURES / "reference-lock.json").read_text())
    for entry in pins["runtime_artifacts"]:
        download(entry, tool_dir)
    launcher = Path(__file__).with_name("RunReference.java")
    subprocess.run([args.java, "-jar", str(tool_dir / "ecj.jar"), "-1.8", "-cp",
                    str(tool_dir / "ij.jar"), "-d", str(tool_dir), str(launcher)],
                   check=True, timeout=120)
    java_version = subprocess.run([args.java, "-version"], check=True,
                                  capture_output=True, text=True).stderr.strip()
    records = {}
    for case in cases:
        name = case["name"]
        if args.case and name not in args.case:
            continue
        directory = work / name
        directory.mkdir(exist_ok=True)
        data = read_input(case)
        source_hash = None if case["kind"] == "binary_mask" else sha(data.tobytes())
        mask_hash = None
        if "mask_zyx" in case:
            mask = np.zeros(data.shape, dtype=np.uint8)
            for z, y, x in case["mask_zyx"]:
                mask[z, y, x] = 1
            mask_hash = sha(mask.tobytes())
            if case["kind"] == "binary_mask":
                data = mask * np.uint8(255)
            mask = place_on_canvas(mask, case)
            tifffile.imwrite(directory / "mask.tif", mask, photometric="minisblack",
                             metadata=None)
        data = place_on_canvas(data, case)
        expanded_case = dict(case, shape_zyx=list(data.shape[:3]))
        input_hash = sha(data.tobytes())
        tifffile.imwrite(directory / "input.tif", data, metadata=None,
                         photometric="rgb" if case["kind"] == "rgb_projection" else "minisblack")
        macro = macro_for(expanded_case, directory, tool_dir)
        macro_path = directory / "run.ijm"
        macro_path.write_text(macro, encoding="utf-8")
        # A failed run must never reuse a previous output as evidence.
        (directory / "output.png").unlink(missing_ok=True)
        cp = os.pathsep.join([str(tool_dir / "ij.jar"), str(tool_dir)])
        invocation = [args.java, "-Xmx2g", f"-Dplugins.dir={tool_dir}", "-cp", cp,
                      "RunReference", str(macro_path)]
        result = subprocess.run(invocation, check=True, capture_output=True,
                                text=True, timeout=180)
        (directory / "stdout.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        output_path = directory / "output.png"
        rgb = np.asarray(Image.open(output_path).convert("RGB"))
        expected_path = FIXTURES / "expected" / (name + ".png")
        if args.record:
            expected_path.parent.mkdir(exist_ok=True)
            shutil.copyfile(output_path, expected_path)
        expected = np.asarray(Image.open(expected_path).convert("RGB"))
        stats = assert_parity(rgb, expected)
        prepared_hash = None
        if case["kind"] != "rgb_projection":
            prepared_hash = sha(tifffile.imread(directory / "prepared.tif").tobytes())
        records[name] = dict(source_raw_sha256=source_hash, input_raw_sha256=input_hash,
                             mask_raw_sha256=mask_hash, prepared_u8_sha256=prepared_hash,
                             output_file_sha256=sha(output_path.read_bytes()),
                             output_rgb_sha256=sha(rgb.tobytes()),
                             output_shape=list(rgb.shape), **stats)
        print(name, json.dumps(stats), flush=True)
    if args.record:
        manifest = dict(schema="nb02-reference-v1", java=java_version,
                        cases_sha256=sha((FIXTURES / "cases.json").read_bytes()),
                        reference_lock_sha256=sha((FIXTURES / "reference-lock.json").read_bytes()),
                        code_hash_normalization="CRLF to LF",
                        launcher_sha256=sha(launcher.read_bytes().replace(b"\r\n", b"\n")),
                        runner_sha256=sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                        records=records)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        for name, record in records.items():
            # PNG container encoding can vary; decoded RGB must be exact.
            for key, value in record.items():
                if key != "output_file_sha256" and manifest["records"][name][key] != value:
                    raise AssertionError(f"{name}: {key} changed")


if __name__ == "__main__":
    main()
