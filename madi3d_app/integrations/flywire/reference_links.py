"""External reference links for a release-qualified FlyWire neuron."""
import re
from urllib.parse import urlencode

import requests

from madi3d_app.io_utils import check_cancelled


def codex_url(context, query=""):
    return codex_search_url(context.dataset, context.segmentation_version, query)


def codex_search_url(dataset, release, query):
    # Browser sign-in stays in Codex. MCNS query routing is unverified.
    if dataset == "mcns":
        return "https://codex.flywire.ai/"
    parameters = {"dataset": dataset, "filter_string": query}
    # Codex chooses the available MANC snapshot.
    if dataset in ("fafb", "banc"):
        parameters["data_version"] = release
    return "https://codex.flywire.ai/app/search?" + urlencode(parameters)


def vfb_instance_url(release, root_id):
    return f"https://virtualflybrain.org/xref/neuronbridge/flywire_fafb:v{release}:{root_id}"


def codex_queries(root_id):
    return (
        ("Open in Codex", root_id),
        ("Upstream in Codex", f"{{upstream}} {root_id}"),
        ("Downstream in Codex", f"{{downstream}} {root_id}"),
    )


_VFB_API = "https://v3-cached.virtualflybrain.org"
_VFB_SITES = {
    ("fafb", "783"): "flywire783",
    ("manc", "1.2.1"): "neuprint_JRC_Manc_1_2_1",
    ("mcns", "0.9"): "male_cns_v0_9",
    ("mcns", "1.0"): "male_cns_v1_0",
    ("banc", "626"): "BANC626",
    ("banc", "888"): "BANC888",
}


def vfb_site(dataset, release):
    return _VFB_SITES.get((dataset, release))


def resolve_vfb_id(site, root_id, cancelled):
    response = requests.get(f"{_VFB_API}/xref", params={"accession": root_id, "db": site}, timeout=5)
    response.raise_for_status()
    ids = {row["id"] for row in response.json()["rows"]
           if row.get("site_id") == site and row.get("accession") == root_id
           and re.fullmatch(r"VFB_[a-zA-Z0-9]+", row.get("id") or "")}
    check_cancelled(cancelled)
    return ids.pop() if len(ids) == 1 else None


def vfb_viewer_url(vfb_id):
    return f"https://v2.virtualflybrain.org/org.geppetto.frontend/geppetto?id={vfb_id}"


def reference_links(site, root_id, cancelled):
    """Resolve the VFB instance and its explicitly listed FlyBase ontology types."""
    vfb_id = resolve_vfb_id(site, root_id, cancelled)
    if vfb_id is None:
        return ()
    response = requests.get(f"{_VFB_API}/get_term_info", params={"id": vfb_id}, timeout=5)
    response.raise_for_status()
    report = response.json()
    check_cancelled(cancelled)
    if report.get("Id") != vfb_id:
        return ()
    links = [("This neuron instance in Virtual Fly Brain", vfb_viewer_url(vfb_id))]
    seen = set()
    types = (report.get("Meta") or {}).get("Types") or ""
    for label, term_id in re.findall(r"\[([^\[\]]+)\]\((FBbt_[0-9]{8})\)", types):
        if term_id in seen:
            continue
        seen.add(term_id)
        links.extend((
            (f"FlyBase - {label}", "https://flybase.org/reports/" + term_id.replace("_", ":")),
            (f"VFB - {label}", "https://virtualflybrain.org/reports/" + term_id),
        ))
    return tuple(links)
