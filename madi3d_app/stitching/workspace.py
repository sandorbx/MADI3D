"""Staged stitching workspace recovery and canonical-result reference projection."""
from __future__ import annotations

import copy
from collections.abc import Mapping

from madi3d_app.operation_status import (
    execution_status as validate_execution_status,
    qc_status as validate_qc_status,
    user_decision as validate_user_decision,
)
from .models import (
    InvalidStitchingWorkspaceInputError,
    STITCHING_JOB_SCHEMA_VERSION,
    STITCHING_WORKSPACE_SCHEMA_VERSION,
    StitchingJob,
    StitchingProjectState,
    StitchingWorkspaceState,
    UNAVAILABLE_STITCHING_RESULT_CODE,
    _LEGACY_STITCHING_JOB_SCHEMAS,
    _LEGACY_STITCHING_WORKSPACE_SCHEMAS,
    _registration_result_payload,
    _workspace_integer,
    _workspace_registration_result,
    _workspace_result_ids,
    _workspace_status,
    materialize_stitching_registration,
    workspace_stitching_settings,
)


_WORKSPACE_RESULT_OVERRIDE_FIELDS = {
    "project_revision",
    "user_decision",
}


def normalize_stitching_registration_operations(
    model, *, tolerate_unavailable=False
):
    """Replace historical full-result payloads with the current canonical form."""
    legacy_workspace_overrides = {}
    for operation_id, raw_record in list(model.operation_records.items()):
        if not isinstance(raw_record, Mapping):
            continue
        if raw_record.get("operation_type") != "stitching_registration":
            continue
        legacy_result = raw_record.get("result", {})
        try:
            if not isinstance(legacy_result, Mapping):
                raise InvalidStitchingWorkspaceInputError(
                    "Historical stitching result must be an object."
                )
            prefix = "stitching-registration:"
            if (
                not isinstance(operation_id, str)
                or not operation_id.startswith(prefix)
                or not operation_id[len(prefix):].strip()
            ):
                raise InvalidStitchingWorkspaceInputError(
                    "Historical stitching operation has an invalid identity."
                )
            candidate = dict(legacy_result)
            candidate.setdefault("registration_id", operation_id[len(prefix):])
            _workspace_registration_result(candidate)
            _workspace_status(
                raw_record.get("execution_status"),
                validate_execution_status,
                "Historical stitching execution status",
                "succeeded",
            )
            _workspace_status(
                raw_record.get("qc_status"),
                validate_qc_status,
                "Historical stitching QC status",
                "not-evaluated",
            )
            if "project_revision" in legacy_result:
                _workspace_integer(
                    legacy_result["project_revision"],
                    "Historical stitching project revision",
                )
            if "user_decision" in legacy_result:
                _workspace_status(
                    legacy_result["user_decision"],
                    validate_user_decision,
                    "Historical stitching result user decision",
                    "unapplied",
                )
            if "user_decision" in raw_record:
                _workspace_status(
                    raw_record["user_decision"],
                    validate_user_decision,
                    "Historical stitching operation user decision",
                    "unapplied",
                )
        except InvalidStitchingWorkspaceInputError:
            if not tolerate_unavailable:
                raise
            model.operation_records.pop(operation_id, None)
            model.unavailable_history_operation_ids.add(operation_id)
            continue
        overrides = {
            key: copy.deepcopy(legacy_result[key])
            for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS
            if key in legacy_result
        }
        if "user_decision" in raw_record:
            overrides.setdefault(
                "user_decision", copy.deepcopy(raw_record["user_decision"])
            )
        if overrides:
            legacy_workspace_overrides[operation_id] = overrides
        try:
            typed = materialize_stitching_registration(raw_record)
        except InvalidStitchingWorkspaceInputError:
            if not tolerate_unavailable:
                raise
            model.operation_records.pop(operation_id, None)
            model.unavailable_history_operation_ids.add(operation_id)
            continue
        record = copy.deepcopy(raw_record)
        record["result"] = _registration_result_payload(typed)
        record["execution_status"] = typed.execution_status
        record["qc_status"] = typed.qc_status
        record.pop("user_decision", None)
        algorithm_version = str(typed.get("algorithm_version") or "").strip()
        if algorithm_version:
            record["algorithm"] = {
                "name": "MADI3D stitching registration",
                "version": algorithm_version,
            }
        model.operation_records[operation_id] = record
    return legacy_workspace_overrides


def project_workspace_results(
    workspace,
    operation_records,
    *,
    restore=False,
    legacy_result_overrides=None,
    unavailable_operation_ids=None,
    tolerate_unavailable=False,
):
    """Project bounded result references and expand them only for live UI use."""

    if not isinstance(workspace, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace must be an object."
        )
    payload = copy.deepcopy(dict(workspace))
    if not payload:
        return {}
    reviews = copy.deepcopy(payload.get("result_reviews", {}))
    if not isinstance(reviews, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace reviews must be an object."
        )
    reviews = dict(reviews)
    for operation_id, review in reviews.items():
        _workspace_result_ids([operation_id], "Stitching workspace review IDs")
        if not isinstance(review, Mapping) or set(review) - {"decision"}:
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace review entries must contain only a decision."
            )
        _workspace_status(
            review.get("decision"), validate_user_decision,
            "Stitching workspace review decision", "unapplied",
        )
    active_project = payload.get("active_project")
    jobs = payload.get("jobs", [])
    if active_project is not None and not isinstance(active_project, Mapping):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching active project must be an object or null."
        )
    if not isinstance(jobs, (list, tuple)) or any(
        not isinstance(owner, Mapping) for owner in jobs
    ):
        raise InvalidStitchingWorkspaceInputError(
            "Stitching workspace jobs must be a list of objects."
        )
    unavailable_operation_ids = (
        unavailable_operation_ids
        if unavailable_operation_ids is not None
        else set()
    )
    owners = [active_project, *jobs]
    for owner in owners:
        if owner is None:
            continue
        result = owner.get("registration_result")
        if result is not None and not isinstance(result, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace registration result must be an object or null."
            )
        result_ids = _workspace_result_ids(
            owner.get("result_operation_ids", []),
            "Stitching workspace result references",
        )
        if result is not None:
            if "operation_id" in result:
                if set(result) - {"operation_id", "result_overrides"}:
                    raise InvalidStitchingWorkspaceInputError(
                        "Compact stitching result references contain unsupported fields."
                    )
                compact_id = result.get("operation_id")
                compact_ids = _workspace_result_ids(
                    [compact_id], "Compact stitching result references"
                )
                result_ids = list(dict.fromkeys([*compact_ids, *result_ids]))
            else:
                _workspace_registration_result(result)
            result_overrides = result.get("result_overrides", {})
            if not isinstance(result_overrides, Mapping):
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching result overrides must be an object."
                )
            for key, value in result_overrides.items():
                if key not in _WORKSPACE_RESULT_OVERRIDE_FIELDS:
                    raise InvalidStitchingWorkspaceInputError(
                        f"Unsupported stitching result override: {key}."
                    )
                if key == "project_revision":
                    _workspace_integer(value, "Stitching project revision")
                else:
                    _workspace_status(
                        value, validate_user_decision,
                        "Stitching result user decision", "unapplied",
                    )
            for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS.intersection(result):
                value = result[key]
                if key == "project_revision":
                    _workspace_integer(value, "Stitching project revision")
                else:
                    _workspace_status(
                        value, validate_user_decision,
                        "Stitching result user decision", "unapplied",
                    )
        if restore:
            legacy_operation_id = str(
                (result or {}).get("operation_id") or ""
            ).strip()
            if legacy_operation_id:
                result_ids = list(dict.fromkeys([
                    legacy_operation_id, *result_ids,
                ]))
            missing_result_ids = [
                operation_id for operation_id in result_ids
                if operation_id not in operation_records
            ]
            undeclared_missing = [
                operation_id for operation_id in missing_result_ids
                if operation_id not in unavailable_operation_ids
            ]
            if undeclared_missing and not tolerate_unavailable:
                shown = undeclared_missing[:5]
                suffix = (
                    f" (+{len(undeclared_missing) - len(shown)} more)"
                    if len(undeclared_missing) > len(shown)
                    else ""
                )
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace references unavailable completed "
                    "results: " + ", ".join(shown) + suffix + "."
                )
            if tolerate_unavailable:
                unavailable_operation_ids.update(missing_result_ids)

            binding_issues = [
                copy.deepcopy(issue)
                for issue in owner.get("binding_issues") or ()
                if isinstance(issue, Mapping)
                and issue.get("code") != UNAVAILABLE_STITCHING_RESULT_CODE
            ]
            if missing_result_ids:
                shown = missing_result_ids[:5]
                suffix = (
                    f" (+{len(missing_result_ids) - len(shown)} more)"
                    if len(missing_result_ids) > len(shown)
                    else ""
                )
                binding_issues.append({
                    "code": UNAVAILABLE_STITCHING_RESULT_CODE,
                    "result_operation_ids": shown,
                    "unavailable_result_count": len(missing_result_ids),
                    "message": (
                        "Historical stitching result unavailable: "
                        + ", ".join(shown)
                        + suffix
                        + ". Recalculate registration or use current visible poses."
                    ),
                })
            owner["binding_issues"] = binding_issues
            registration_operation_id = next(
                (
                    operation_id for operation_id in result_ids
                    if operation_records.get(operation_id, {}).get("operation_type")
                    == "stitching_registration"
                ),
                "",
            )
            if not registration_operation_id:
                owner.pop("registration_result", None)
                owner["result_operation_ids"] = list(dict.fromkeys(result_ids))
                continue
            record = operation_records.get(registration_operation_id)
            overrides = copy.deepcopy(
                (legacy_result_overrides or {}).get(registration_operation_id, {})
            )
            if result:
                overrides.update(copy.deepcopy(result.get("result_overrides") or {}))
                overrides.update({
                    key: copy.deepcopy(result[key])
                    for key in _WORKSPACE_RESULT_OVERRIDE_FIELDS
                    if key in result
                })
            unsupported = sorted(set(overrides) - _WORKSPACE_RESULT_OVERRIDE_FIELDS)
            if unsupported:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace registration reference contains unsupported "
                    "overrides: " + ", ".join(unsupported) + "."
                )
            expanded = materialize_stitching_registration(record).to_dict()
            expanded.update(overrides)
            review = reviews.get(registration_operation_id) or {}
            if review:
                expanded["user_decision"] = validate_user_decision(
                    review.get("decision", "unapplied")
                )
            owner["registration_result"] = expanded
            owner["result_operation_ids"] = list(dict.fromkeys(result_ids or [
                registration_operation_id
            ]))
            continue

        if result and result.get("registration_id"):
            operation_id = "stitching-registration:" + str(result["registration_id"])
            record = operation_records.get(operation_id)
            if record is None:
                raise InvalidStitchingWorkspaceInputError(
                    "Stitching workspace result has no canonical registration operation."
                )
            result_ids.insert(0, operation_id)
            decision = validate_user_decision(
                result.get("user_decision", "unapplied")
            )
            if decision != "unapplied":
                prior = reviews.get(operation_id)
                review = {"decision": decision}
                if prior is not None and prior != review:
                    raise InvalidStitchingWorkspaceInputError(
                        "Stitching workspace contains conflicting review state for "
                        f"{operation_id}."
                    )
                reviews[operation_id] = review
        result_ids = list(dict.fromkeys(result_ids))
        missing_result_ids = [
            operation_id for operation_id in result_ids
            if operation_id not in operation_records
            and operation_id not in unavailable_operation_ids
        ]
        if missing_result_ids:
            shown = missing_result_ids[:5]
            suffix = (
                f" (+{len(missing_result_ids) - len(shown)} more)"
                if len(missing_result_ids) > len(shown)
                else ""
            )
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace references unavailable completed "
                "results: " + ", ".join(shown) + suffix + "."
            )
        owner["result_operation_ids"] = result_ids
        owner["settings"] = workspace_stitching_settings(owner.get("settings"))
        if owner.get("execution_status") == "running":
            owner["execution_status"] = "interrupted"
        owner.pop("registration_result", None)
        for field_name in (
            "binding_issues",
            "created",
            "execution_phase",
            "loaded_job_id",
            "outputs",
            "qc_status",
            "selection_descriptors",
            "user_decision",
        ):
            owner.pop(field_name, None)
    payload["schema"] = STITCHING_WORKSPACE_SCHEMA_VERSION
    for job in payload.get("jobs", []):
        job["schema"] = STITCHING_JOB_SCHEMA_VERSION
    payload["result_reviews"] = reviews
    return payload


def stage_stitching_project_state(raw, volume_sources, *, report_loss):
    """Project recoverable workspace state against available result owners."""
    legacy_result_overrides = normalize_stitching_registration_operations(
        volume_sources, tolerate_unavailable=True
    )
    if raw is None or (isinstance(raw, Mapping) and not raw):
        return {}
    if not isinstance(raw, Mapping):
        report_loss(
            "invalid-stitching-root", "stitching_workspace",
            "The stitching workspace was omitted; completed scientific results remain available.",
        )
        return {}
    schema = raw.get("schema")
    if schema not in {
        STITCHING_WORKSPACE_SCHEMA_VERSION, *_LEGACY_STITCHING_WORKSPACE_SCHEMAS
    }:
        report_loss(
            "invalid-stitching-schema", "stitching_workspace.schema",
            "The stitching workspace was omitted; completed scientific results remain available.",
        )
        return {}
    if set(raw) - {
        "schema", "active_project", "jobs", "job_counter", "result_reviews",
        "release_temporary_inputs", "load_completed_outputs",
    }:
        report_loss(
            "unsupported-stitching-fields", "stitching_workspace",
            "The stitching workspace was omitted because it contains unsupported fields.",
        )
        return {}
    raw_reviews = raw.get("result_reviews", {})
    if not isinstance(raw_reviews, Mapping):
        report_loss(
            "invalid-stitching-reviews", "stitching_workspace.result_reviews",
            "The stitching workspace was omitted because its review state was malformed.",
        )
        return {}

    raw_jobs = raw.get("jobs", ())
    if not isinstance(raw_jobs, (list, tuple)):
        raw_jobs = ()
        report_loss(
            "invalid-stitching-job-list", "stitching_workspace.jobs",
            "Queued stitching jobs were omitted because their list shape was invalid.",
        )

    def project_owner(owner, *, active_project):
        if not isinstance(owner, Mapping):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching workspace owner must be an object."
            )
        if not active_project and owner.get("schema") not in (
            None,
            STITCHING_JOB_SCHEMA_VERSION,
            *_LEGACY_STITCHING_JOB_SCHEMAS,
        ):
            raise InvalidStitchingWorkspaceInputError(
                "Stitching job schema is unsupported."
            )
        projected_workspace = project_workspace_results(
            {
                "schema": schema,
                "active_project": owner if active_project else None,
                "jobs": [] if active_project else [owner],
                "job_counter": raw.get("job_counter", 0),
                "result_reviews": {},
            },
            volume_sources.operation_records,
            restore=True,
            legacy_result_overrides=legacy_result_overrides,
            unavailable_operation_ids=set(
                volume_sources.unavailable_history_operation_ids
            ),
            tolerate_unavailable=True,
        )
        projected_owner = (
            projected_workspace.get("active_project")
            if active_project else projected_workspace["jobs"][0]
        )
        if active_project:
            StitchingProjectState.from_dict(projected_owner)
        else:
            validated_job = StitchingJob.from_dict(projected_owner)
            projected_owner["execution_status"] = validated_job.execution_status
        return projected_owner

    active = None
    if raw.get("active_project") is not None:
        try:
            active = project_owner(raw["active_project"], active_project=True)
        except InvalidStitchingWorkspaceInputError:
            report_loss(
                "invalid-stitching-active-project",
                "stitching_workspace.active_project",
                "The active stitching project was omitted; valid queued jobs were retained.",
            )
    jobs = []
    for index, value in enumerate(raw_jobs):
        try:
            jobs.append(project_owner(value, active_project=False))
        except InvalidStitchingWorkspaceInputError:
            report_loss(
                "invalid-stitching-job", f"stitching_workspace.jobs[{index}]",
                "This queued stitching job was omitted; valid siblings were retained.",
            )

    candidate = {
        "schema": schema,
        "active_project": active,
        "jobs": jobs,
        "job_counter": raw.get("job_counter", 0),
        "result_reviews": copy.deepcopy(dict(raw_reviews)),
        "release_temporary_inputs": raw.get("release_temporary_inputs", True),
        "load_completed_outputs": raw.get("load_completed_outputs", False),
    }
    try:
        projected = project_workspace_results(
            candidate,
            volume_sources.operation_records,
            restore=True,
            legacy_result_overrides=legacy_result_overrides,
            unavailable_operation_ids=set(
                volume_sources.unavailable_history_operation_ids
            ),
            tolerate_unavailable=True,
        )
        normalized_state = StitchingWorkspaceState.from_dict(projected)
        normalized = projected if not normalized_state.is_empty else {}
    except InvalidStitchingWorkspaceInputError:
        report_loss(
            "invalid-stitching-relationships", "stitching_workspace",
            "The stitching workspace was omitted because result references, reviews, or queued-job relationships could not be reconciled.",
        )
        return {}

    available_operations = volume_sources.operation_records
    for owner_name, owner in (
        [("active_project", normalized.get("active_project"))]
        + [
            (f"jobs[{index}]", job)
            for index, job in enumerate(normalized.get("jobs", ()))
        ]
    ):
        if owner is None:
            continue
        for operation_id in owner.get("result_operation_ids", ()):
            if operation_id not in available_operations:
                report_loss(
                    "unavailable-stitching-result",
                    f"stitching_workspace.{owner_name}.result_operation_ids",
                    "The saved registration result was unavailable; its reference and current project state were retained.",
                    approval_required=False,
                    source_protection_required=False,
                )
    return normalized
