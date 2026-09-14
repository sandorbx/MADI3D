"""Persistent operation references, never a second scientific source registry."""
WORKSPACE_KEY = "neuronbridge_workspace"


def source_key(reference):
    return reference["channel_id"] if reference["kind"] == "volume" else "scene:" + reference["object_id"]


def validate_workspace(metadata):
    workspace = metadata.get(WORKSPACE_KEY)
    if workspace is None:
        return
    if not isinstance(workspace, dict) or workspace.get("version") != 1 or not isinstance(workspace.get("sources"), list):
        raise ValueError("Invalid NeuronBridge source workspace.")
    seen = set()
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
