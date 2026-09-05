# -*- coding: utf-8 -*-
"""Durable result-bundle helpers for MADI3D registration.

This module is deliberately independent from Qt, VTK, ITK, and CMTK execution.
It owns only output-policy validation, deterministic result-folder layout,
manifest bookkeeping, small QC summaries, and ordinary file copies.  Image and
mesh serialization stays with MADI3D's central Save As writers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


RESULT_LAYOUTS = {"separate", "common"}
VOLUME_FORMAT_EXTENSIONS = {
    "nrrd": ".nrrd",
    "nifti": ".nii.gz",
    "tiff": ".tif",
    "h5j": ".h5j",
}
MESH_FORMAT_EXTENSIONS = {
    "vtp": ".vtp",
    "obj": ".obj",
    "vtk": ".vtk",
}

# Keep registration paths usable by Windows APIs and third-party tools that do
# not reliably support extended-length paths. Count UTF-16 code units because
# that is the relevant Windows representation even when tests run elsewhere.
WINDOWS_SAFE_PATH_BUDGET = 240
WINDOWS_FILESYSTEM_COMPONENT_BUDGET = 255
_PATH_HASH_LENGTH = 12
_PATH_HASH_SEPARATOR = "--"
_MIN_COMPACTED_COMPONENT_UNITS = (
    1 + len(_PATH_HASH_SEPARATOR) + _PATH_HASH_LENGTH
)
_TEMPORARY_SUFFIX_RESERVE = ".partial"
_LONGEST_ARTIFACT_EXTENSION = ".registration.json"
_SEPARATE_CATEGORY_RESERVE = max(
    len("transforms"),
    len("reformatted"),
    len("qc"),
    len("logs"),
    _MIN_COMPACTED_COMPONENT_UNITS,
)

DEFAULT_RESULT_OUTPUT = {
    "enabled": False,
    "root_dir": "",
    "layout": "separate",
    "write_transforms": True,
    "write_reformatted": True,
    "write_qc_summary": True,
    "write_qc_volumes": False,
    "write_log": True,
    "volume_format": "nrrd",
    "mesh_format": "vtp",
}


class RegistrationOutputError(RuntimeError):
    pass


def safe_output_stem(value: Any, fallback: str = "registration") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip(" ._")
    return text or str(fallback or "registration")


def _windows_text_units(value: Any) -> int:
    return len(str(value).encode("utf-16-le")) // 2


def _truncate_windows_text(value: str, max_units: int) -> str:
    kept = []
    used = 0
    for character in str(value):
        units = _windows_text_units(character)
        if used + units > int(max_units):
            break
        kept.append(character)
        used += units
    return "".join(kept)


def bounded_output_component(
    value: Any,
    max_units: int,
    fallback: str = "registration",
    *,
    hash_source: Any | None = None,
) -> str:
    """Return a sanitized component bounded for deterministic Windows paths."""
    original = str(value or "")
    sanitized = safe_output_stem(original, fallback)
    max_units = min(int(max_units), WINDOWS_FILESYSTEM_COMPONENT_BUDGET)
    if _windows_text_units(sanitized) <= max_units:
        return sanitized

    digest_source = original if hash_source is None else str(hash_source)
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[
        :_PATH_HASH_LENGTH
    ]
    hashed_suffix = _PATH_HASH_SEPARATOR + digest
    prefix_units = max_units - _windows_text_units(hashed_suffix)
    if prefix_units < 1:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "collision-safe registration output name."
        )
    prefix = _truncate_windows_text(sanitized, prefix_units).rstrip(" ._")
    if not prefix:
        prefix = _truncate_windows_text(safe_output_stem(fallback), prefix_units)
    if not prefix:
        raise RegistrationOutputError(
            "The available filesystem-component budget is too small for a "
            "readable registration output name."
        )
    return prefix + hashed_suffix


def _short_result_root_error(root: Path) -> RegistrationOutputError:
    return RegistrationOutputError(
        "The registration result directory is too long to create Windows-compatible "
        f"result paths within the {WINDOWS_SAFE_PATH_BUDGET}-character budget. "
        f"Choose a shorter registration result directory. Current directory: {root}"
    )


def _run_component_capacity(root: Path, layout: str) -> int:
    trailing_units = (
        1
        + _MIN_COMPACTED_COMPONENT_UNITS
        + _windows_text_units(_LONGEST_ARTIFACT_EXTENSION)
        + _windows_text_units(_TEMPORARY_SUFFIX_RESERVE)
    )
    if str(layout) == "separate":
        trailing_units += 1 + _SEPARATE_CATEGORY_RESERVE
    return min(
        WINDOWS_FILESYSTEM_COMPONENT_BUDGET,
        WINDOWS_SAFE_PATH_BUDGET
        - _windows_text_units(root)
        - 1
        - trailing_units,
    )


def _require_bundle_root_capacity(root: Path, layout: str) -> int:
    capacity = _run_component_capacity(root, layout)
    if capacity < _MIN_COMPACTED_COMPONENT_UNITS:
        raise _short_result_root_error(root)
    return capacity


def _assert_windows_safe_path(path: Path, *, extra_suffix: str = "") -> None:
    resolved = Path(path).resolve()
    if (
        _windows_text_units(resolved) + _windows_text_units(extra_suffix)
        > WINDOWS_SAFE_PATH_BUDGET
    ):
        raise _short_result_root_error(resolved.parent)


def normalize_result_output_policy(
    value: dict | None,
    *,
    force_enabled: bool = False,
    require_root: bool = False,
) -> dict:
    raw = dict(DEFAULT_RESULT_OUTPUT)
    raw.update(dict(value or {}))
    policy = {
        "enabled": bool(raw.get("enabled") or force_enabled),
        "root_dir": os.path.abspath(os.path.expanduser(str(raw.get("root_dir") or "").strip()))
        if str(raw.get("root_dir") or "").strip() else "",
        "layout": str(raw.get("layout") or "separate").lower(),
        "write_transforms": bool(raw.get("write_transforms", True)),
        "write_reformatted": bool(raw.get("write_reformatted", True)),
        "write_qc_summary": bool(raw.get("write_qc_summary", True)),
        "write_qc_volumes": bool(raw.get("write_qc_volumes", False)),
        "write_log": bool(raw.get("write_log", True)),
        "volume_format": str(raw.get("volume_format") or "nrrd").lower(),
        "mesh_format": str(raw.get("mesh_format") or "vtp").lower(),
    }
    if policy["layout"] not in RESULT_LAYOUTS:
        raise RegistrationOutputError(
            f"Unknown registration result-folder layout: {policy['layout']!r}."
        )
    if policy["volume_format"] not in VOLUME_FORMAT_EXTENSIONS:
        raise RegistrationOutputError(
            f"Unsupported registration volume output format: {policy['volume_format']!r}."
        )
    if policy["mesh_format"] not in MESH_FORMAT_EXTENSIONS:
        raise RegistrationOutputError(
            f"Unsupported registration mesh output format: {policy['mesh_format']!r}."
        )
    if policy["enabled"] and not any(
        policy[key]
        for key in (
            "write_transforms", "write_reformatted", "write_qc_summary",
            "write_qc_volumes", "write_log",
        )
    ):
        raise RegistrationOutputError(
            "Result writing is enabled, but no result artifact is selected."
        )
    if (require_root or policy["enabled"]) and not policy["root_dir"]:
        raise RegistrationOutputError("Choose a registration result output folder first.")
    return policy


def ensure_output_root_writable(policy: dict) -> Path:
    """Create and verify the configured output root without creating a run bundle."""
    normalized = normalize_result_output_policy(policy, require_root=True)
    root = Path(normalized["root_dir"]).resolve()
    _require_bundle_root_capacity(root, normalized["layout"])
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RegistrationOutputError(f"Registration output path is not a directory: {root}")
    probe = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=".madi3d-registration-write-", suffix=".tmp",
            dir=root, delete=False,
        )
        probe = Path(handle.name)
        handle.write(b"MADI3D registration output write test\n")
        handle.close()
    except Exception as exc:
        raise RegistrationOutputError(
            f"Registration output folder is not writable: {root}\n{exc}"
        ) from exc
    finally:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
    return root


def partial_output_path(path: os.PathLike | str) -> Path:
    """Return a same-format temporary path suitable for atomic replacement."""
    path = Path(path)
    name = path.name
    if name.lower().endswith(".nii.gz"):
        temporary = path.with_name(name[:-7] + ".partial.nii.gz")
    else:
        temporary = path.with_name(path.stem + ".partial" + path.suffix)
    _assert_windows_safe_path(temporary)
    return temporary


def volume_extension(output_format: str) -> str:
    try:
        return VOLUME_FORMAT_EXTENSIONS[str(output_format).lower()]
    except KeyError as exc:
        raise RegistrationOutputError(
            f"Unsupported registration volume output format: {output_format!r}."
        ) from exc


def mesh_extension(output_format: str) -> str:
    try:
        return MESH_FORMAT_EXTENSIONS[str(output_format).lower()]
    except KeyError as exc:
        raise RegistrationOutputError(
            f"Unsupported registration mesh output format: {output_format!r}."
        ) from exc


def registration_qc_summary(chain_payload: dict) -> dict:
    chain = copy.deepcopy(dict(chain_payload or {}))
    summaries = []
    warnings = []
    for stage in chain.get("stages") or []:
        stage = dict(stage or {})
        details = dict(stage.get("details") or {})
        summary = {
            "name": str(stage.get("name") or "Stage"),
            "kind": str(stage.get("kind") or "unknown"),
            "execution_status": str(stage.get("execution_status") or "pending"),
            "qc_status": str(stage.get("qc_status") or "not-evaluated"),
            "user_decision": str(stage.get("user_decision") or "unapplied"),
            "metric_value": stage.get("metric_value"),
            "ncc": stage.get("ncc"),
            "iterations": int(stage.get("iterations") or 0),
            "stop_condition": str(stage.get("stop_condition") or ""),
        }
        for key in (
            "support_overlap_fraction",
            "requested_support_overlap_fraction",
            "automatic_center_fallback",
            "linear_sanity",
            "deformation_qc",
        ):
            if key in details:
                summary[key] = copy.deepcopy(details.get(key))
        execution_error = str(details.get("execution_error") or details.get("artifact_error") or "").strip()
        if execution_error:
            summary["execution_error"] = execution_error
            warnings.append(f"{summary['name']}: {execution_error}")
        for item in details.get("qc_failures") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(f"{summary['name']}: {text}")
        for item in details.get("quality_warnings") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(f"{summary['name']}: {text}")
        deformation_qc = dict(details.get("deformation_qc") or {})
        for item in deformation_qc.get("warnings") or []:
            text = str(item or "").strip()
            if text:
                warnings.append(text)
        summaries.append(summary)
    # Preserve order while removing repeated warnings from stage/model mirrors.
    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "format": "MADI3D Registration QC Summary",
        "format_version": 1,
        "registration_id": str(chain.get("registration_id") or ""),
        "source_name": str(chain.get("source_name") or ""),
        "target_name": str(chain.get("target_name") or ""),
        "created": str(chain.get("created") or ""),
        "has_deformation": bool(chain.get("has_deformation", False)),
        "execution_status": str(chain.get("execution_status") or "pending"),
        "qc_status": str(chain.get("qc_status") or "not-evaluated"),
        "user_decision": str(chain.get("user_decision") or "unapplied"),
        "stage_path": [
            str(stage.get("kind") or "unknown")
            for stage in chain.get("stages") or []
            if str(stage.get("execution_status") or "pending") == "succeeded"
        ],
        "stages": summaries,
        "warnings": unique_warnings,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _assert_windows_safe_path(path)
    _assert_windows_safe_path(temporary)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class RegistrationOutputBundle:
    run_dir: Path
    policy: dict
    run_id: str
    label: str
    queued: bool = False
    job_id: str = ""
    registration_algorithm_version: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        policy: dict,
        *,
        label: str,
        run_id: str,
        queued: bool = False,
        job_id: str = "",
        registration_algorithm_version: str = "",
    ) -> "RegistrationOutputBundle":
        normalized = normalize_result_output_policy(
            policy, force_enabled=bool(queued), require_root=True
        )
        root = Path(normalized["root_dir"]).resolve()
        run_capacity = _require_bundle_root_capacity(root, normalized["layout"])
        root.mkdir(parents=True, exist_ok=True)
        token = safe_output_stem(run_id, "run")[:12]
        folder_candidate = f"{safe_output_stem(label, 'registration')}--{token}"
        folder_name = bounded_output_component(
            folder_candidate,
            run_capacity,
            "registration",
            hash_source=f"{str(label or 'Registration')}--{str(run_id)}",
        )
        run_dir = root / folder_name
        if run_dir.exists():
            raise RegistrationOutputError(
                f"Registration result folder already exists; refusing to overwrite it: {run_dir}"
            )
        run_dir.mkdir(parents=False, exist_ok=False)
        bundle = cls(
            run_dir=run_dir,
            policy=normalized,
            run_id=str(run_id),
            label=str(label or "Registration"),
            queued=bool(queued),
            job_id=str(job_id or ""),
            registration_algorithm_version=str(registration_algorithm_version or ""),
        )
        bundle.manifest = {
            "format": "MADI3D Registration Result Bundle",
            "format_version": 1,
            "registration_algorithm_version": bundle.registration_algorithm_version,
            "run_id": bundle.run_id,
            "job_id": bundle.job_id or None,
            "queued": bundle.queued,
            "label": bundle.label,
            "created": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "policy": copy.deepcopy(normalized),
            "artifacts": [],
            "warnings": [],
            "errors": [],
        }
        # Store the output root only as contextual metadata; all artifact paths
        # remain relative so the completed bundle can be moved as one directory.
        bundle.manifest["policy"]["root_dir"] = "."
        bundle.write_manifest()
        return bundle

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def directory(self, category: str) -> Path:
        if self.policy["layout"] == "separate":
            trailing_units = (
                1
                + _MIN_COMPACTED_COMPONENT_UNITS
                + _windows_text_units(_LONGEST_ARTIFACT_EXTENSION)
                + _windows_text_units(_TEMPORARY_SUFFIX_RESERVE)
            )
            category_capacity = min(
                WINDOWS_FILESYSTEM_COMPONENT_BUDGET,
                WINDOWS_SAFE_PATH_BUDGET
                - _windows_text_units(self.run_dir.resolve())
                - 1
                - trailing_units,
            )
            if category_capacity < _MIN_COMPACTED_COMPONENT_UNITS:
                raise _short_result_root_error(Path(self.policy["root_dir"]))
            category_candidate = safe_output_stem(category, "results").lower()
            component = bounded_output_component(
                category_candidate,
                category_capacity,
                "results",
                hash_source=category,
            )
            path = self.run_dir / component
        else:
            path = self.run_dir
        _assert_windows_safe_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, category: str, stem: str, extension: str) -> Path:
        extension = str(extension or "")
        if extension and not extension.startswith("."):
            extension = "." + extension
        parent = self.directory(category)
        stem_capacity = min(
            WINDOWS_FILESYSTEM_COMPONENT_BUDGET
            - _windows_text_units(extension)
            - _windows_text_units(_TEMPORARY_SUFFIX_RESERVE),
            WINDOWS_SAFE_PATH_BUDGET
            - _windows_text_units(parent.resolve())
            - 1
            - _windows_text_units(extension)
            - _windows_text_units(_TEMPORARY_SUFFIX_RESERVE),
        )
        if stem_capacity < _MIN_COMPACTED_COMPONENT_UNITS:
            raise _short_result_root_error(Path(self.policy["root_dir"]))
        filename = bounded_output_component(stem, stem_capacity) + extension
        path = parent / filename
        _assert_windows_safe_path(path, extra_suffix=_TEMPORARY_SUFFIX_RESERVE)
        return path

    def relative(self, path: os.PathLike | str) -> str:
        candidate = Path(path).resolve()
        root = self.run_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise RegistrationOutputError(
                f"Result artifact is outside its registration bundle: {candidate}"
            )
        return candidate.relative_to(root).as_posix()

    def record_artifact(self, kind: str, path: os.PathLike | str, **metadata) -> None:
        entry = {
            "kind": str(kind or "artifact"),
            "path": self.relative(path),
        }
        entry.update({str(k): copy.deepcopy(v) for k, v in metadata.items() if v is not None})
        self.manifest.setdefault("artifacts", []).append(entry)
        self.write_manifest()

    def add_warning(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text not in self.manifest.setdefault("warnings", []):
            self.manifest["warnings"].append(text)
            self.write_manifest()

    def add_error(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            self.manifest.setdefault("errors", []).append(text)
            self.write_manifest()

    def write_manifest(self, status: str | None = None) -> None:
        if status:
            self.manifest["status"] = str(status)
        self.manifest["updated"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(self.manifest_path, self.manifest)

    def write_qc_summary(self, chain_payload: dict, stem: str) -> Path:
        path = self.artifact_path("qc", stem, ".qc.json")
        _atomic_json(path, registration_qc_summary(chain_payload))
        self.record_artifact(
            "qc_summary",
            path,
            registration_id=str((chain_payload or {}).get("registration_id") or ""),
        )
        return path

    def copy_qc_volume(self, source: os.PathLike | str, stem: str, role: str) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise RegistrationOutputError(f"Registration QC volume is missing: {source_path}")
        destination = self.artifact_path("qc", stem, ".nrrd")
        shutil.copy2(source_path, destination)
        self.record_artifact("qc_volume", destination, role=str(role or "qc"))
        return destination

    def write_log(self, text: str) -> Path:
        path = self.artifact_path("logs", "registration", ".log")
        path.write_text(str(text or ""), encoding="utf-8")
        self.record_artifact("run_log", path)
        return path

    def finalize(self, status: str = "complete") -> Path:
        self.write_manifest(status=status)
        return self.run_dir


__all__ = [
    "DEFAULT_RESULT_OUTPUT",
    "MESH_FORMAT_EXTENSIONS",
    "RESULT_LAYOUTS",
    "RegistrationOutputBundle",
    "RegistrationOutputError",
    "VOLUME_FORMAT_EXTENSIONS",
    "WINDOWS_SAFE_PATH_BUDGET",
    "bounded_output_component",
    "ensure_output_root_writable",
    "mesh_extension",
    "normalize_result_output_policy",
    "partial_output_path",
    "registration_qc_summary",
    "safe_output_stem",
    "volume_extension",
]
