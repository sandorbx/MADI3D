# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the detached FlyWire backend using generated inputs or opt-in live retrieval."""

import argparse
from dataclasses import asdict
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile


def smoke_offline(scratch):
    import DracoPy
    import numpy as np
    import pyarrow as pa
    import pyarrow.feather as feather
    from cloudvolume import Mesh, Skeleton
    from madi3d_app.integrations.flywire import backend, catalog_download, catalog_review, input_parser
    from madi3d_app.integrations.flywire.catalog import Catalog, CatalogProduct
    from madi3d_app.integrations.flywire.geometry import verify_artifact, write_mesh, write_skeleton
    from madi3d_app.integrations.flywire.records import ReleaseContext

    assert importlib.metadata.version("cloud-volume") == backend.BACKEND_VERSION
    assert callable(catalog_download.download_and_install)
    source = scratch / "classification.csv.gz"
    root_id = "18446744073709551615"
    source.write_bytes(gzip.compress(f"root_id,class\n{root_id},Synthetic motor\n9,Other\n".encode(), mtime=0))
    catalog = Catalog(scratch / "fafb.sqlite")
    catalog.install([CatalogProduct(source, "classification")])
    inputs = input_parser.parse_root_ids(root_id, context=catalog.context)
    assert catalog_review.enrich_inputs(inputs, catalog).retrieval_root_ids == (root_id,)
    assert catalog.search("motor", {"class": "SYNTHETIC MOTOR"}).root_ids == (root_id,)
    assert catalog.search(filters={"class": "motor"}).root_ids == ()

    # Generated int64 exercises the Arrow decoder without redistributing rows.
    body = 9007199254740993
    source = scratch / "annotation.feather"
    feather.write_feather(pa.table({"bodyId": pa.array([body], type=pa.int64()),
                                    "type": ["synthetic"]}), source)
    catalog = Catalog(scratch / "mcns.sqlite", context=ReleaseContext("mcns", "1.0"))
    catalog.install([CatalogProduct(source, "body_annotations")])
    assert catalog.search(filters={"cell_type": "synthetic"}).root_ids == (str(body),)

    # Nonzero origin, distinct extents and measured radii expose unit mistakes.
    points = np.array([[1000, 2000, 3000], [5000, 2000, 3000], [1000, 9000, 3000]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    decoded = DracoPy.decode(DracoPy.encode(points, faces, preserve_order=True))
    mesh = Mesh(decoded.points, decoded.faces)
    vtp = scratch / "mesh.vtp"
    evidence = write_mesh(vtp, mesh.vertices, mesh.faces)
    assert (evidence["vertices"], evidence["primitives"]) == (3, 1)
    np.testing.assert_allclose(evidence["bounds_um"], [1, 2, 3, 5, 9, 3], atol=1e-3)
    verify_artifact(vtp, "mesh", evidence)

    edges = np.array([[0, 1]], dtype=np.uint32)
    skeleton = Skeleton(points[:2], edges, radii=np.array([500, 1000], dtype=np.float32))
    swc = scratch / "geometry.swc"
    evidence = write_skeleton(swc, skeleton.vertices, skeleton.edges, skeleton.radius, np.eye(4)[:3])
    assert (evidence["vertices"], evidence["primitives"], evidence["components"]) == (2, 1, 1)
    np.testing.assert_allclose(evidence["bounds_um"], [1, 2, 3, 5, 2, 3])
    assert swc.read_text(encoding="ascii").splitlines() == ["1 0 1 2 3 0.5 -1", "2 0 5 2 3 1 1"]
    verify_artifact(swc, "swc", evidence)


def smoke_live(scratch, dataset):
    from madi3d_app.integrations.flywire.geometry import GeometryLimits, verify_artifact
    from madi3d_app.integrations.flywire.process import run_bounded
    from madi3d_app.integrations.flywire.records import FlyWireRemoteDescriptor, ReleaseContext

    release, root = {
        "fafb": ("783", "720575940619118427"),
        "manc": ("1.2.1", "10001"),
        "mcns": ("1.0", "100000"),
    }[dataset]
    context = ReleaseContext(dataset, release)
    for representation in context.profile.representations:
        directory = scratch / representation
        directory.mkdir()
        descriptor = FlyWireRemoteDescriptor(
            context, root, representation, 0 if representation == "mesh" else None,
        )
        limits = GeometryLimits(max_payload_bytes=256 * 1024**2, max_artifact_bytes=256 * 1024**2)
        request = directory / "request.json"
        request.write_text(json.dumps({"descriptor": descriptor.to_dict(), "limits": asdict(limits)}),
                           encoding="utf-8")
        run = run_bounded(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", str(request)],
            deadline_seconds=120, max_rss_bytes=2 * 1024**3,
        )
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        if run["status"] != "ok" or result.get("status") != "ok":
            raise RuntimeError(f"{dataset} {representation} failed: {run} {result}")
        artifact = directory / ("geometry.vtp" if representation == "mesh" else "geometry.swc")
        verify_artifact(artifact, representation, result["evidence"], limits)
        print(f"{dataset} {release} {representation} verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", choices=("fafb", "manc", "mcns"), help="Contact one public source")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    if args.worker:
        from madi3d_app.integrations.flywire.process import worker_entry
        return worker_entry(args.worker, 0)

    with tempfile.TemporaryDirectory(prefix="flywire-smoke-") as directory:
        scratch = Path(directory)
        smoke_offline(scratch)
        print("Offline catalog, Draco mesh and SWC smoke passed")
        if args.live:
            smoke_live(scratch, args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
