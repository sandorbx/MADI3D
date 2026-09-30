"""Persistent operation references, never a second scientific source registry."""
import copy

WORKSPACE_KEY = "neuronbridge_workspace"


def source_key(reference):
    return reference["channel_id"] if reference["kind"] == "volume" else "scene:" + reference["object_id"]


def link_auto_query(metadata, key, query_id):
    """Make a completed MIP the current automatic image for one source."""
    from .query import ASSOCIATION_KEY, QUERY_KEY

    result = dict(metadata)
    workspace = dict(result.get(WORKSPACE_KEY, {}))
    references = copy.deepcopy(workspace.get("sources", []))
    reference = next((ref for ref in references if source_key(ref) == key), None)
    if reference is None:
        raise ValueError("The NeuronBridge source was removed before its image completed.")
    previous = reference.get("auto_query_id")
    reference["auto_query_id"] = query_id
    workspace["sources"] = references
    result[WORKSPACE_KEY] = workspace
    if previous and previous != query_id and previous not in result.get(ASSOCIATION_KEY, {}).values():
        registry = dict(result.get(QUERY_KEY, {}))
        registry.pop(previous, None)
        result[QUERY_KEY] = registry
    return result


def remove_sources_and_mips(metadata, keys):
    """Drop selected workspace sources and their automatic MIP evidence."""
    from .query import ASSOCIATION_KEY, QUERY_KEY, query_dependencies

    keys = set(keys)
    result = dict(metadata)
    workspace = dict(result.get(WORKSPACE_KEY, {}))
    references = copy.deepcopy(workspace.get("sources", []))
    removed = [ref for ref in references if source_key(ref) in keys]
    retained = [ref for ref in references if source_key(ref) not in keys]
    if not removed:
        return result
    registry = dict(result.get(QUERY_KEY, {}))
    delete_ids = {ref["auto_query_id"] for ref in removed if ref.get("auto_query_id")}
    # A signal's automatic MIP may also consume a separately listed mask.
    for ref in retained:
        query_id = ref.get("auto_query_id")
        record = registry.get(query_id)
        if record is not None and any(
            dependency["selection"]["channel"]["selector"] in keys
            for dependency in query_dependencies(record)
        ):
            delete_ids.add(query_id)
            ref.pop("auto_query_id", None)
    for query_id in delete_ids:
        registry.pop(query_id, None)
    workspace["sources"] = retained
    result[WORKSPACE_KEY] = workspace
    if delete_ids:
        result[QUERY_KEY] = registry
        result[ASSOCIATION_KEY] = {
            session_id: query_id
            for session_id, query_id in result.get(ASSOCIATION_KEY, {}).items()
            if query_id not in delete_ids
        }
    return result


def validate_workspace(metadata):
    from .query import QUERY_KEY, query_dependencies

    workspace = metadata.get(WORKSPACE_KEY)
    if workspace is None:
        return
    if not isinstance(workspace, dict) or workspace.get("version") != 1 or not isinstance(workspace.get("sources"), list):
        raise ValueError("Invalid NeuronBridge source workspace.")
    seen = set()
    linked = set()
    for reference in workspace["sources"]:
        if not isinstance(reference, dict) or reference.get("kind") not in {"volume", "geometry"}:
            raise ValueError("Invalid NeuronBridge source reference.")
        fields = ("acquisition_id", "channel_id") if reference["kind"] == "volume" else ("object_id",)
        if any(not isinstance(reference.get(key), str) or not reference[key] for key in fields):
            raise ValueError("NeuronBridge sources require stable identities.")
        if reference.get("role") not in ({"signal", "mask"} if reference["kind"] == "volume" else {"geometry"}):
            raise ValueError("Invalid NeuronBridge source role.")
        if type(reference.get("frame_index")) is not int or reference["frame_index"] < 0:
            raise ValueError("Invalid NeuronBridge source frame.")
        key = source_key(reference)
        if key in seen:
            raise ValueError("Duplicate NeuronBridge workspace source.")
        seen.add(key)
        query_id = reference.get("auto_query_id")
        if query_id is not None:
            record = metadata.get(QUERY_KEY, {}).get(query_id)
            if (not isinstance(query_id, str) or not query_id or query_id in linked
                    or record is None or not any(
                        dependency["selection"]["channel"]["selector"] == key
                        for dependency in query_dependencies(record)
                    )):
                raise ValueError("Invalid NeuronBridge automatic image reference.")
            linked.add(query_id)
