"""Synthetic offline install/restart/search check for all frozen build targets.

The caller supplies an isolated temporary directory. This never accesses a
public collection, configured library location, credentials, or the network.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .cdm_records import CDMGeometry, CDMMapping, CDMParameters, CDMSelection, TEMPLATE_ID
from .local_catalog import IMAGE_BUCKET, ObjectMissing, libraries_from_catalog, member_url
from .local_library import LibraryManager, decoder_check
from .local_search import search_library
from .query import QueryInput, QueryLaunch, complete_query
from .records import ChannelSelection, SearchResults, SourceIdentity


def check(root):
    import tifffile
    from madi3d_app.volume.geometry import VolumeWorkingGrid
    decoder_check()
    root = Path(root) / "Local search Ω with spaces"
    root.mkdir()
    pixels = np.full((1, 1, 2), 255, np.uint8)
    grid = VolumeWorkingGrid(dimensions=(2, 1, 1), spacing=(.5189161, .5189161, 1), origin=(0, 0, 0),
        direction=np.eye(3), physical_units=("um",) * 3, source_coordinate_space_id="synthetic-smoke",
        coordinate_mode="working-grid", geometry_basis="partially-assumed", physical_grid_state="incomplete-geometry",
        assumed_fields=("origin",), warnings=("Synthetic package test; no biological alignment.",))
    selection = CDMSelection("smoke", "1", SourceIdentity(kind="lm", library="synthetic fixture"), ChannelSelection("0", 0), 0, "ZYX")
    transform = np.eye(4)
    transform[:3, 3] = (400, 140, 50)
    launch = QueryLaunch.capture(QueryInput(selection, CDMGeometry(grid, "1", np.eye(4)), pixels), None,
        CDMMapping(transform, "synthetic-smoke"), CDMParameters(allow_exploratory=True))
    result, query = complete_query(launch, root / "retained query", "smoke-query", "Synthetic package fixture")
    config = {"anatomicalAreas": {"Brain": {"alignmentSpace": TEMPLATE_ID}}, "stores": {
        "fl:open_data:brain": {"anatomicalArea": "Brain", "prefixes": {"CDM": f"https://s3.amazonaws.com/{IMAGE_BUCKET}/"},
        "customSearch": {"searchFolder": "searchable_neurons", "emLibraries": [
            {"name": "FlyEM_Hemibrain_v1.2.1", "publishedNamePrefix": "hemibrain:v1.2.1", "count": 1}]}}}}
    library = libraries_from_catalog({"data_version": "v0_0", "config": {"payload": config}, "synthetic_smoke_fixture": True})[0]
    key = f"{TEMPLATE_ID}/{library.name}/searchable_neurons/0/fixture-{TEMPLATE_ID}-CDM.tif"
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, result.rgb, photometric="rgb", compression="lzw")
    objects = {library.manifest_url: json.dumps([key]).encode(), member_url(library, key): buffer.getvalue()}

    class SyntheticTransport:
        def download(self, url, path, *, max_bytes):
            if url not in objects:
                raise ObjectMissing("Synthetic metadata is deliberately unresolved.")
            data = objects[url]
            if len(data) > max_bytes:
                raise ValueError("Synthetic object exceeds its bounded transfer limit.")
            Path(path).write_bytes(data)
            return {"url": url, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "hash_authority": "synthetic_fixture", "retrieved_at": "2026-09-12T00:00:00+00:00"}

    manager = LibraryManager(root / "persistent data")
    snapshot = manager.create(library)
    manager.install(snapshot, transport=SyntheticTransport())
    reopened = LibraryManager(manager.root)
    matches = search_library(reopened, snapshot, json.loads(json.dumps(query)))
    evidence = matches.session.context
    if evidence["counts"] != {"total": 1, "examined": 1, "failed": 0, "matched": 1, "retained": 1}:
        raise ValueError("Frozen local search did not examine and retain its synthetic candidate.")
    if matches.occurrences[0].matched_pixels != 2:
        raise ValueError("Frozen local positive-CDS count differs from its known fixture.")
    saved = json.loads(json.dumps(matches.to_dict()))
    reopened.remove(snapshot)
    if SearchResults.from_dict(saved) != matches:
        raise ValueError("Local results did not survive removal of their library.")
    # Exercise the same persistence and standalone paths in every frozen target.
    from madi3d_app.project.document import ProjectDocument, ProjectObjectRecord, ProjectObjectType
    from madi3d_app.project.save_controller import publish_project_document
    from madi3d_app.project.package_io import read_validated_madi3d_package
    from madi3d_app.project.scientific_records import (
        export_provenance_projection, write_object_metadata_sidecar, read_object_metadata_sidecar,
    )
    from .evidence import SESSION_KEY, object_metadata_for_match
    from .query import QUERY_KEY, query_png
    document = ProjectDocument(objects=(ProjectObjectRecord(1, 0, ProjectObjectType.REMOTE, "Synthetic hit",
        object_metadata=object_metadata_for_match(matches, matches.occurrences[0])),),
        project_metadata={SESSION_KEY: {matches.session.session_id: saved}, QUERY_KEY: {query["query_id"]: query}})
    package = root / "provenance.madi3d"
    publish_project_document(package, document)
    del document
    restored = read_validated_madi3d_package(package)
    if query_png(restored.project_metadata[QUERY_KEY][query["query_id"]]) != query_png(query):
        raise ValueError("Packaged query pixels changed across save/reload.")
    geometry = root / "synthetic.obj"
    geometry.write_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    projection = export_provenance_projection(restored.objects[0].object_metadata, restored.volume_sources,
                                              restored.project_metadata)
    write_object_metadata_sidecar(geometry, projection)
    fresh = ProjectDocument(objects=(ProjectObjectRecord(1, 0, ProjectObjectType.MESH, "Reimported",
        object_metadata=read_object_metadata_sidecar(geometry)),))
    publish_project_document(package, fresh)
    del fresh
    final = read_validated_madi3d_package(package)
    occurrence = next(iter(final.neuronbridge_index.matches.values())).occurrence
    if occurrence != matches.occurrences[0]:
        raise ValueError("Standalone reimport changed historical numerical search evidence.")
