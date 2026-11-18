"""Run the pinned EM/ImageJ producer and original scorer for NB-GRAD-GEN.

This is an opt-in capture; routine tests consume the committed PNGs and manifest.
The upstream checkouts and Maven dependencies stay outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np

PRODUCER_PIN = "83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2"
SCORER_PIN = "8b85c458f3b9231ec7bff5f1b4262cbf2abf8c3d"
IMAGEJ_SOURCE_PIN = "v1.53v"
IMAGEJ_COMMIT = "72c46b115c91cce887793141cb43a8ec31c82f5d"
SCORER_TESTS = ["Shape2DMatchColorDepthSearchAlgorithmTest", "GradientAreaGapUtilsTest"]
ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests/fixtures/neuronbridge_gradient"
VNC_FIXTURES = ROOT / "tests/fixtures/neuronbridge_vnc"
PRODUCER_SOURCE = "Gradient_conv_multithread.java"
IMAGE_RESOURCE = "colormipsearch-api/src/test/resources/colormipsearch/api/cdsearch/"

# The extra-clear characterization binds to the production contract rather than
# restating it: the same annotation boxes the pinned producer translates and the
# same VNC positive-score exclusions the local scorer executes.
sys.path.insert(0, str(ROOT))
from madi3d_app.integrations.neuronbridge.gradient_preparation import (  # noqa: E402
    ANNOTATION_LEFT_BOX, ANNOTATION_RIGHT_BOX)
from madi3d_app.integrations.neuronbridge.search_profiles import VNC  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def decoded(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.ndim == 2:
        if pixels.dtype.kind != "u" or pixels.dtype.itemsize != 2:
            raise ValueError(f"Expected unsigned 16-bit gray PNG, got {pixels.dtype} from {path}")
        canonical = np.ascontiguousarray(pixels.astype("<u2", copy=False))
        encoding = "unsigned uint16 little endian, row-major"
    else:
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 image, got {pixels.dtype}/{pixels.shape} from {path}")
        canonical = np.ascontiguousarray(pixels.astype("u1", copy=False))
        encoding = "RGB uint8, row-major"
    return canonical, encoding


def normalized_source_sha(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".java", ".py"}:
        data = data.replace(b"\r\n", b"\n")
    return sha(data)


def vnc_annotation_clear_masks(canvas_yx: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Producer-clear and positive-exclusion masks on the VNC canvas, derived
    from the production contract (Gradient_conv_multithread L161-173 translated
    as ``ANNOTATION_*_BOX``; VNC profile ``score_exclusions``)."""
    height, width = canvas_yx
    producer = np.zeros((height, width), bool)
    rows, columns = ANNOTATION_RIGHT_BOX
    producer[:min(rows, height), max(0, width - columns):width] = True
    rows, columns = ANNOTATION_LEFT_BOX
    producer[:min(rows, height), 0:min(columns, width)] = True
    positive = np.zeros((height, width), bool)
    for anchor, x, y, excluded_w, excluded_h in VNC.score_exclusions:
        left = x if anchor == "left" else width - x - excluded_w
        positive[max(0, y):min(height, y + excluded_h),
                 max(0, left):min(width, left + excluded_w)] = True
    return producer, positive


def vnc_extra_clear_signal(pixels: np.ndarray) -> dict:
    """Real candidate signal inside extra_clear = producer clear AND NOT positive
    exclusion. Candidate foreground = RGB pixels with any nonzero channel."""
    producer, positive = vnc_annotation_clear_masks(pixels.shape[:2])
    extra = producer & ~positive
    foreground = np.any(pixels != 0, axis=2)
    region = pixels[extra]
    nonzero_in_extra = int(np.count_nonzero(foreground & extra))
    return {
        "foreground_definition": "candidate foreground = RGB pixels with any nonzero channel",
        "foreground_pixels": int(np.count_nonzero(foreground)),
        "nonzero_pixels_in_extra_clear": nonzero_in_extra,
        "rgb_sum_in_extra_clear": int(region.sum()),
        "rgb_maximum_channel_in_extra_clear": int(region.max()) if region.size else 0,
        "extra_clear_foreground_fraction": (
            nonzero_in_extra / int(np.count_nonzero(foreground))
            if foreground.any() else 0.0),
    }


def checked(args: list[str | Path], cwd: Path, env: dict[str, str], timeout: int = 900) -> bytes:
    try:
        result = subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True,
                                capture_output=True, timeout=timeout)
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as error:
        sys.stderr.buffer.write(error.stdout + error.stderr)
        raise


def png_row(path: Path, output_path: str, input_path: str, input_sha: str,
            input_pixels_sha: str, label_policy: str, radius: int, **extra) -> dict:
    pixels, encoding = decoded(path)
    return {
        "path": output_path,
        "file_sha256": file_sha(path),
        "decoded_sha256": sha(pixels.tobytes()),
        "shape": list(pixels.shape),
        "dtype": "<u2" if pixels.ndim == 2 else "|u1",
        "pixel_encoding": encoding,
        "input": {"path": input_path, "file_sha256": input_sha,
                  "decoded_sha256": input_pixels_sha},
        "label_policy": label_policy,
        "radius": radius,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-upstream", type=Path, required=True)
    parser.add_argument("--scorer-upstream", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--imagej-jar", type=Path, required=True)
    parser.add_argument("--imagej-license", type=Path, required=True)
    parser.add_argument("--imagej-source", type=Path, required=True)
    parser.add_argument("--record", action="store_true", help="replace the retained capture after validation")
    args = parser.parse_args()
    producer = args.producer_upstream.resolve()
    scorer = args.scorer_upstream.resolve()
    java_home = args.java_home.resolve()
    imagej_jar = args.imagej_jar.resolve()
    imagej_license = args.imagej_license.resolve()
    imagej_source = args.imagej_source.resolve()
    java = java_home / "bin" / ("java.exe" if os.name == "nt" else "java")
    javac = java_home / "bin" / ("javac.exe" if os.name == "nt" else "javac")
    env = dict(os.environ, JAVA_HOME=str(java_home))
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")

    producer_head = checked(["git", "rev-parse", "HEAD"], producer, env).decode().strip()
    if producer_head != PRODUCER_PIN:
        raise ValueError(f"Expected producer {PRODUCER_PIN}, got {producer_head}")
    if checked(["git", "status", "--porcelain"], producer, env).strip():
        raise ValueError("Producer checkout must be clean")
    scorer_head = checked(["git", "rev-parse", "HEAD"], scorer, env).decode().strip()
    if scorer_head != SCORER_PIN:
        raise ValueError(f"Expected scorer {SCORER_PIN}, got {scorer_head}")
    if checked(["git", "status", "--porcelain"], scorer, env).strip():
        raise ValueError("Scorer checkout must be clean")
    imagej_head = checked(["git", "rev-parse", "HEAD"], imagej_source, env).decode().strip()
    if imagej_head != IMAGEJ_COMMIT:
        raise ValueError(f"Expected ImageJ {IMAGEJ_SOURCE_PIN} source {IMAGEJ_COMMIT}, got {imagej_head}")
    pinned_imagej_license = checked(["git", "show", f"{IMAGEJ_COMMIT}:LICENSE.txt"], imagej_source, env)
    if imagej_license.read_bytes() != pinned_imagej_license:
        raise ValueError("Retained ImageJ license differs from the pinned source")
    pinned_producer_license = checked(["git", "cat-file", "blob", f"{PRODUCER_PIN}:LICENSE"], producer, env)
    retained_producer_license = ROOT / "LICENSES/EM_MIP_search-LICENSE.txt"
    if retained_producer_license.read_bytes() != pinned_producer_license:
        raise ValueError("Retained producer license differs from the pinned source blob")

    cases = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    lm_names = sorted({Path(c["lm"]).name for c in cases})
    lm_paths = {name: FIXTURES / "inputs/lms" / name for name in lm_names}
    # Bind the build and resulting observations to the exact harness bytes used
    # before compilation. Re-check immediately before publishing a capture so a
    # concurrent edit cannot label old bytecode with a newer source hash.
    harness_files = [REFERENCE / "RunProducer.java", REFERENCE / "WriteZGapImages.java",
                     REFERENCE / "reproduce_producer.py", REFERENCE / "RunReference.java",
                     REFERENCE / "GradientProbes.java"]
    harness_receipt = {p.relative_to(ROOT).as_posix(): normalized_source_sha(p)
                       for p in harness_files}

    scorer_wrapper = scorer / ("mvnw.cmd" if os.name == "nt" else "mvnw")
    maven_command = [scorer_wrapper, "-pl", "colormipsearch-api", "-am",
                     "-Dtest=" + ",".join(SCORER_TESTS),
                     "-Dsurefire.failIfNoSpecifiedTests=false", "test"]
    scratch_parent = scorer / "colormipsearch-api/target"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nb-grad-gen-", dir=scratch_parent) as tmp:
        work = Path(tmp)
        maven_output = checked(maven_command, scorer, env)
        reports = scorer / "colormipsearch-api/target/surefire-reports"
        suites = [ET.parse(reports / f"TEST-org.janelia.colormipsearch.cds.{name}.xml").getroot()
                  for name in SCORER_TESTS]
        if any(int(s.attrib.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped")):
            raise ValueError("Pinned scorer tests failed or skipped")
        properties = {p.attrib["name"]: p.attrib["value"]
                      for p in suites[0].findall("properties/property")}
        classpath = properties["java.class.path"]

        producer_classes = work / "producer-classes"
        producer_classes.mkdir()
        producer_java = REFERENCE / "RunProducer.java"
        producer_command = [javac, "-cp", imagej_jar, "-d", producer_classes,
                            producer / PRODUCER_SOURCE, producer_java]
        checked(producer_command, producer, env)
        producer_output = work / "producer-output"
        producer_output.mkdir()
        producer_classpath = os.pathsep.join((str(producer_classes), str(imagej_jar)))
        producer_run_command = [java, "-cp", producer_classpath, "RunProducer",
                                str(FIXTURES / "inputs/lms"), str(work), str(producer_output),
                                str(VNC_FIXTURES)]
        checked(producer_run_command, producer, env)

        # Build score inputs with the observed producer outputs in place of archived gradients.
        score_inputs = work / "score-inputs"
        shutil.copytree(FIXTURES / "inputs", score_inputs)
        generated_grad_dir = score_inputs / "generated"
        generated_grad_dir.mkdir()
        for lm_name in lm_names:
            stem = Path(lm_name).stem
            candidate = producer_output / f"{stem}.png"
            if not candidate.exists():
                raise ValueError(f"Missing generated gradient for {stem}")
            shutil.copyfile(candidate, generated_grad_dir / f"{stem}.png")
        generated_cases = [{**c, "gradient": f"generated/{Path(c['lm']).stem}.png", "zgap": None}
                           for c in cases]
        cases_path = work / "generated-cases.json"
        cases_path.write_text(json.dumps(generated_cases, indent=2) + "\n", encoding="utf-8")
        java_sources = [REFERENCE / "RunReference.java", REFERENCE / "GradientProbes.java",
                        REFERENCE / "WriteZGapImages.java"]
        score_classes = work / "score-classes"
        score_classes.mkdir()
        checked([javac, "-cp", classpath, "-d", score_classes, *java_sources], scorer, env)
        zgap_tmp = work / "zgap"
        score_classpath = os.pathsep.join((str(score_classes), classpath))
        zgap_command = [java, "-Djava.awt.headless=true", "-cp", score_classpath,
                        "org.janelia.colormipsearch.cds.WriteZGapImages",
                        str(score_inputs), str(cases_path), str(zgap_tmp)]
        checked(zgap_command, scorer, env)
        score_json = work / "generated-scores.json"
        score_command = [java, "-Djava.awt.headless=true", "-cp", score_classpath,
                         "org.janelia.colormipsearch.cds.RunReference",
                         str(score_inputs), str(cases_path), str(score_json)]
        checked(score_command, scorer, env)
        generated_observations = json.loads(score_json.read_text(encoding="utf-8"))

        producer_dir = FIXTURES / "producer"
        zgap_dir = producer_dir / "zgap"
        producer_dir.mkdir(parents=True, exist_ok=True)
        zgap_dir.mkdir(parents=True, exist_ok=True)
        output_rows = []
        for lm_name, input_path in lm_paths.items():
            stem = Path(lm_name).stem
            output = producer_output / f"{stem}.png"
            retained = producer_dir / output.name
            if args.record:
                shutil.copyfile(output, retained)
            elif not retained.exists() or not np.array_equal(decoded(output)[0], decoded(retained)[0]):
                raise ValueError(f"Original producer decoded pixels differ from retained PNG: {retained.name}")
            inp = input_path
            input_pixels, _ = decoded(inp)
            row = png_row(retained, retained.relative_to(ROOT).as_posix(),
                          inp.relative_to(ROOT).as_posix(), file_sha(inp), sha(input_pixels.tobytes()),
                          "reference-labels", 20, kind="real-lm", profile="Brain")
            row["input"]["shape"] = list(input_pixels.shape)
            archive = FIXTURES / "inputs/grad" / f"{stem}.png"
            archived_pixels, _ = decoded(archive)
            generated_pixels, _ = decoded(retained)
            row["archived_gradient"] = {
                "path": archive.relative_to(ROOT).as_posix(),
                "file_sha256": file_sha(archive),
                "decoded_sha256": sha(archived_pixels.tobytes()),
                "pixel_equal": bool(np.array_equal(generated_pixels, archived_pixels)),
                "different_pixels": int(np.count_nonzero(generated_pixels != archived_pixels)),
                "maximum_absolute_difference": int(np.max(np.abs(generated_pixels.astype(np.int32) - archived_pixels.astype(np.int32)))),
            }
            output_rows.append({"id": stem, **row})

        # Retained real MANC (VNC) TIFFs: the same pinned producer path on real
        # VNC canvases, with the annotation-clear/positive-exclusion difference
        # characterized against the exact retained pixels.
        vnc_input_paths = sorted(VNC_FIXTURES.glob("*.tif"))
        if len(vnc_input_paths) != 3:
            raise ValueError(f"Expected three retained MANC TIFFs, found {len(vnc_input_paths)}")
        for input_path in vnc_input_paths:
            stem = input_path.stem
            output = producer_output / f"{stem}.png"
            if not output.exists():
                raise ValueError(f"Missing VNC producer output {stem}.png")
            retained = producer_dir / output.name
            if args.record:
                shutil.copyfile(output, retained)
            elif not retained.exists() or not np.array_equal(decoded(output)[0], decoded(retained)[0]):
                raise ValueError(f"Original producer decoded pixels differ from retained PNG: {output.name}")
            input_pixels, _ = decoded(input_path)
            row = png_row(retained, retained.relative_to(ROOT).as_posix(),
                          input_path.relative_to(ROOT).as_posix(), file_sha(input_path),
                          sha(input_pixels.tobytes()), "reference-labels", 20,
                          kind="real-vnc-manc", profile="VNC")
            row["input"]["shape"] = list(input_pixels.shape)
            row["vnc_extra_clear_signal"] = vnc_extra_clear_signal(input_pixels)
            output_rows.append({"id": stem, **row})

        white = [255, 255, 255]
        synthetic_sources = {
            "empty": {"recipe": {"background_rgb": [0, 0, 0], "points": []}, "profile": "Brain", "shape": [566, 1210, 3]},
            "full": {"recipe": {"fill_rgb": white}, "profile": "Brain", "shape": [566, 1210, 3]},
            "isolated": {"recipe": {"background_rgb": [0, 0, 0], "points": [{"xy": [605, 283], "rgb": white}]}, "profile": "Brain", "shape": [566, 1210, 3]},
            "border": {"recipe": {"background_rgb": [0, 0, 0], "points": [{"xy": [600, 0], "rgb": white}, {"xy": [600, 565], "rgb": white}]}, "profile": "Brain", "shape": [566, 1210, 3]},
            "colors": {"recipe": {"background_rgb": [0, 0, 0], "points": [
                {"xy": [360, 280], "rgb": [255, 0, 0]}, {"xy": [460, 280], "rgb": [0, 255, 0]},
                {"xy": [560, 280], "rgb": [0, 0, 255]}, {"xy": [660, 280], "rgb": white},
                {"xy": [760, 280], "rgb": [1, 0, 0]}, {"xy": [860, 280], "rgb": [2, 0, 0]}]},
                "profile": "Brain", "shape": [566, 1210, 3]},
            "labels": {"recipe": {"background_rgb": [0, 0, 0], "points": [
                {"xy": [100, 50], "rgb": white}, {"xy": [1110, 50], "rgb": white},
                {"xy": [200, 104], "rgb": white}, {"xy": [1060, 99], "rgb": white},
                {"xy": [400, 105], "rgb": white}, {"xy": [605, 283], "rgb": white}]},
                "profile": "Brain", "shape": [566, 1210, 3]},
            "vnc-isolated": {"recipe": {"background_rgb": [0, 0, 0], "points": [{"xy": [286, 604], "rgb": white}]},
                             "profile": "VNC", "shape": [1209, 573, 3]},
        }
        input_files = {
            "synthetic-empty.png": "empty.tif", "synthetic-full.png": "full.tif",
            "synthetic-isolated.png": "isolated.tif", "synthetic-border.png": "border.tif",
            "synthetic-colors.png": "colors.tif", "synthetic-labels.png": "labels.tif",
            "vnc-canvas.png": "vnc-isolated.tif",
            "synthetic-labels-no-labels.png": "labels.tif",
        }
        for output_name, input_name in input_files.items():
            output = producer_output / output_name
            if not output.exists():
                raise ValueError(f"Missing synthetic producer output {output_name}")
            retained = producer_dir / output_name
            if args.record:
                shutil.copyfile(output, retained)
            elif not retained.exists() or not np.array_equal(decoded(output)[0], decoded(retained)[0]):
                raise ValueError(f"Original producer decoded pixels differ from retained PNG: {output_name}")
            temporary_input = work / "synthetic-inputs" / input_name
            source_kind = input_name.removesuffix(".tif")
            policy = "no-labels" if output_name.endswith("-no-labels.png") else "reference-labels"
            source_info = synthetic_sources[source_kind]
            recipe = source_info["recipe"]
            profile = source_info["profile"]
            shape = source_info["shape"]
            if policy == "no-labels":
                recipe = {**recipe, "label_clearing": "bypassed explicitly"}
                profile = "Brain"
            temp_pixels, _ = decoded(temporary_input)
            row = png_row(retained, retained.relative_to(ROOT).as_posix(),
                          f"synthetic://{output_name.removesuffix('.png')}", file_sha(temporary_input),
                          sha(temp_pixels.tobytes()), policy, 20,
                          kind="synthetic", profile=profile, recipe=recipe,
                          input_shape=shape, input_dtype="|u1",
                          input_format="RGB uint8; ImageJ ColorProcessor saved as TIFF")
            output_rows.append({"id": output_name.removesuffix(".png"), **row})

        zgap_rows = []
        for path in sorted(zgap_tmp.glob("*.png")):
            retained = zgap_dir / path.name
            if args.record:
                shutil.copyfile(path, retained)
            elif not retained.exists() or not np.array_equal(decoded(path)[0], decoded(retained)[0]):
                raise ValueError(f"Original Java Z-gap pixels differ from retained PNG: {path.name}")
            pixels, encoding = decoded(retained)
            source_name = path.name.removesuffix("-no-labels.png").removesuffix(".png") + ".tif"
            input_path = FIXTURES / "inputs/lms" / source_name
            input_pixels, _ = decoded(input_path)
            zgap_rows.append({"id": path.name.removesuffix(".png"),
                              "path": retained.relative_to(ROOT).as_posix(),
                              "file_sha256": file_sha(retained),
                              "decoded_sha256": sha(pixels.tobytes()),
                              "shape": list(pixels.shape), "dtype": "|u1",
                              "pixel_encoding": encoding,
                              "input_path": input_path.relative_to(ROOT).as_posix(),
                              "input_file_sha256": file_sha(input_path),
                              "input_decoded_sha256": sha(input_pixels.tobytes()),
                              "input_shape": list(input_pixels.shape),
                              "label_policy": "no-labels" if path.stem.endswith("-no-labels") else "reference-labels",
                              "preparation": "identity or ImageTestUtils labels -> ColorTransformation.mask(20) -> unsafeMaxFilter(10) -> toImageArray"})

        score_rows = generated_observations["real_cases"]
        baseline_rows = json.loads((FIXTURES / "observations.json").read_text(encoding="utf-8"))["real_cases"]
        baselines_by_case = {row["case"]: row for row in baseline_rows}
        scores = []
        for score in score_rows:
            baseline = baselines_by_case[score["case"]]
            keys = ("gradient_area_gap", "high_expression_area", "combined_long", "score_int", "mirrored")
            scores.append({"id": score["case"], "combination": score["combination"],
                           **{key: score[key] for key in keys},
                           "zgap_mode": score["zgap_mode"],
                           "zgap_decoded_sha256": score["zgap_rgb_sha256"],
                           "baseline_archived_gradient_score": {key: baseline[key] for key in keys},
                           "same_as_archived_gradient_score": all(score[key] == baseline[key] for key in keys)})

        source_paths = [PRODUCER_SOURCE, "README.md", "LICENSE"]
        source_hashes = {p: sha(checked(["git", "cat-file", "blob", f"{PRODUCER_PIN}:{p}"], producer, env))
                         for p in source_paths}
        producer_clear, positive_exclusion = vnc_annotation_clear_masks(VNC.search_canvas_yx)
        extra_clear = producer_clear & ~positive_exclusion
        runner_files = [imagej_license,
                        ROOT / "LICENSES/EM_MIP_search-LICENSE.txt",
                        ROOT / "LICENSES/ImageJ-LICENSE.txt"]
        runner_hashes = {p.relative_to(ROOT).as_posix(): normalized_source_sha(p)
                         for p in runner_files if p.is_file()}
        runner_hashes.update(harness_receipt)
        manifest = {
            "schema": 1,
            "evidence": "observed original Java/ImageJ execution",
            "producer": {"repository": "https://github.com/JaneliaSciComp/EM_MIP_search",
                         "revision": producer_head, "source_sha256": source_hashes,
                         "method": "Gradient_conv_multithread.Threfunction and gradientslice",
                         "settings": {"ImageJ": IMAGEJ_SOURCE_PIN, "radius": 20,
                                      "max_radius_dialog": "displayed value is never read by run(); stored Prefs default is 20",
                                      "threads": 1,
                                      "rgb_weights": [1.0 / 3, 1.0 / 3, 1.0 / 3],
                                      "gray_conversion_scaling": True,
                                      "weighted_color_preference": False,
                                      "inverting_lut": False,
                                      "producer_labels": {"left": {"xy": [0, 0], "size": [330, 105]},
                                                          "right": {"anchor": "right", "x_offset": -270,
                                                                    "y": 0, "size": [270, 100]}},
                                      "color_to_gray": "ImageConverter.convertToGray8",
                                      "maximum_filter": "ImageJ Maximum... circular radius 20",
                                      "gray16_mask": "values below 1 -> 65535, remaining -> 0",
                                      "distance": "original four-connected outward propagation; scan x then y; unsigned 16-bit output"}},
            "imagej": {"version": IMAGEJ_SOURCE_PIN, "source_revision": imagej_head,
                       "source_license_sha256": sha(pinned_imagej_license),
                       "jar_sha256": file_sha(imagej_jar),
                       "license_source": "ImageJ v1.53v LICENSE.txt", "license_sha256": file_sha(imagej_license),
                       "license_path": (ROOT / "LICENSES/ImageJ-LICENSE.txt").relative_to(ROOT).as_posix()},
            "scorer": {"repository": "https://github.com/JaneliaSciComp/colormipsearch",
                       "revision": scorer_head, "original_tests": SCORER_TESTS,
                       "reference_manifest_sha256": file_sha(FIXTURES / "manifest.json"),
                       "reference_harness_sha256": {name: harness_receipt[(REFERENCE / name).relative_to(ROOT).as_posix()]
                                                     for name in ("RunReference.java", "GradientProbes.java")},
                       "original_junit_failures": 0, "original_junit_errors": 0,
                       "generated_variant": "RunReference on original EM/LM cases with generated gradient PNGs and Java-derived ZGapImage; supplied Z-gap bypassed explicitly"},
            "execution": {"java_version": checked([java, "-version"], ROOT, env).decode(errors="replace"),
                          "os": os.name,
                          "producer_compile_command": ["javac", "-cp", "<ImageJ 1.53v JAR>", "-d", "<scratch>",
                                                       PRODUCER_SOURCE, "RunProducer.java"],
                          "producer_run_command": ["java", "-cp", "<producer classes and ImageJ 1.53v JAR>",
                                                   "RunProducer", "<retained LM inputs>", "<scratch>",
                                                   "<scratch output>", "<retained VNC inputs>"],
                          "scorer_test_command": [scorer_wrapper.name, *[str(x) for x in maven_command[1:]]],
                          "zgap_command": ["java", "-Djava.awt.headless=true", "-cp",
                                           "<compiled harness and Surefire classpath>",
                                           "org.janelia.colormipsearch.cds.WriteZGapImages", "<score inputs>",
                                           "<generated cases with zgap=null>", "<Java Z-gap output directory>"],
                          "generated_scorer_command": ["java", "-Djava.awt.headless=true", "-cp",
                                                       "<compiled harness and Surefire classpath>",
                                                       "org.janelia.colormipsearch.cds.RunReference",
                                                       "<generated score inputs>",
                                                       "<generated cases with zgap=null>", "<Java output JSON>"],
                          "runner_sha256": runner_hashes,
                          "dependency_sha256": {imagej_jar.name: file_sha(imagej_jar)}},
            "outputs": output_rows,
            "z_gap_outputs": zgap_rows,
            "generated_scores": scores,
            "vnc_annotation_clear": {
                "canvas_yx": list(VNC.search_canvas_yx),
                "content_offset_y": VNC.content_offset_yx[0],
                "producer_left_box_rows_x_columns": list(ANNOTATION_LEFT_BOX),
                "producer_right_box_rows_x_columns": list(ANNOTATION_RIGHT_BOX),
                "positive_score_exclusions": [list(entry) for entry in VNC.score_exclusions],
                "producer_clear_pixels": int(np.count_nonzero(producer_clear)),
                "positive_exclusion_pixels": int(np.count_nonzero(positive_exclusion)),
                "extra_clear_pixels": int(np.count_nonzero(extra_clear)),
                "definition": "extra_clear = producer annotation clear AND NOT VNC positive-score exclusion; the pinned producer erases its fixed brain-origin boxes on every canvas while VNC positive CDS excludes only its header rectangles"},
            "limitations": ["No source canvas large enough to exceed unsigned-16-bit distance range was executed; supported Brain/VNC canvases have finite maximum grid path distance below 65535.",
                            "Producer outputs differing from archived fixtures are distinct generated variants; no identity claim is made.",
                            "The per-MANC extra-clear signal characterization covers only the retained three-member real MANC subset; it does not establish that every MANC or VNC candidate has no signal inside that region."],
        }
        manifest_path = FIXTURES / "producer_manifest.json"
        serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        current_harness = {p.relative_to(ROOT).as_posix(): normalized_source_sha(p)
                           for p in harness_files}
        if current_harness != harness_receipt:
            raise ValueError("Harness source changed during capture; rerun against stable source bytes")
        if args.record:
            manifest_path.write_text(serialized, encoding="utf-8", newline="\n")
        elif not manifest_path.exists():
            raise ValueError("No retained producer manifest; use --record after review")
        else:
            retained = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("producer", "imagej", "scorer", "outputs", "z_gap_outputs",
                        "generated_scores", "vnc_annotation_clear"):
                if manifest[key] != retained.get(key):
                    raise ValueError(f"Observed Java producer evidence differs from retained manifest: {key}")
        print(f"Captured {len(output_rows)} gradient outputs, {len(zgap_rows)} Z-gap outputs, {len(scores)} Java score rows")
        print(f"Maven output bytes: {len(maven_output)}")


if __name__ == "__main__":
    main()
