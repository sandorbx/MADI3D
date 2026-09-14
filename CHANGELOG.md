# MADI3D changelog

## 0.32.0 — release preparation, 2026-09-13

### Added

- Native Color-Depth MIP generation from signal, masks and masked original signal, with Brain and VNC profiles, previews, image import/export and retained query evidence.
- Offline positive Color-Depth search against installed Hemibrain v1.2.1, FlyWire FAFB v783 realigned and MANC v1.2.1 libraries, with explicit resumable installation and immutable library snapshots.
- NeuronBridge public precomputed-result lookup, lossless CSV import, thumbnail browsing, selective candidate loading and query/result persistence for later 3D comparison.
- Automatic 3D stitching overlap discovery with coarse-to-fine retries and retained ambiguity.
- Configurable registration QC severity and thresholds, recorded separately from operation completion.
- Synchronized camera, clipping, translation and segmentation controls.

### Improved

- Background segmentation calculations, bounded save/export work and cancellable bulk loading.
- Native VTK SWC tessellation and atomic surface caching; loaded SWC export keeps the GUI responsive.
- TIFF memory estimates, volume display-limit handling, scene-tree removal and surface-export memory bounds.
- Metadata updates reuse checked NeuronBridge evidence. Changed images, dependencies, sessions and cross-references remain validated; project reload validates serialized evidence again.
- EM CSV sex-label conflicts no longer prevent NeuronBridge thumbnails or SWC/OBJ retrieval. Original CSV and public metadata remain separate, with a saved discrepancy visible in retrieval feedback, result details and Object Info. Hard image-identity conflicts and LM acquisition selectors still block retrieval.
- SWC queries explicitly retain their experimental rendered-surface input, radius multiplier and tessellation provenance. Existing query evidence is preserved.
- README and website feature descriptions, example workflow and NeuronBridge screenshot.

### Release process and scope

- Qt/PySide6/Shiboken6 LGPL notices, complete license texts, explicit replacement/debugging rights, and source/build instructions now accompany the packages. Publication requires exact corresponding source archives hosted with each release and validates the legal files and replaceable bindings in all four archives. Store/MSIX packaging is gated pending a separate LGPL review.

- The Publish workflow is the only entry point for four-platform validation. It gates packaging and public synchronization on scientific and offline NeuronBridge regressions, then requires all four packages and frozen smoke checks.
- Standalone scientific, CMTK and package workflows select one platform. Preparation does not launch GitHub Actions or publish a release.
- Local positive-CDS scores are distinct from hosted scores; authenticated hosted custom-query submission is not included. Full-library performance and biological retrieval accuracy are not established. SWC queries are rendered-surface projections, not canonical skeleton queries.

## 0.31.3 — 2026-09-11

- Segmentation extraction preserves the complete source grid for selected voxels and original signal. Outside-mask voxels are zero; original-signal extraction retains its configured mask margin.
- Updated extraction guidance and microscopy import/decode documentation.

## 0.31.2 — 2026-09-10

- Managed FFmpeg downloads use operating-system HTTPS trust, with pinned release and checksum validation.
- Added managed native CMTK installation for Apple Silicon, with cancellation, rollback and retained backend selection.
- Corrected CAVE color-update pacing and connection lifecycle handling.
- Added frozen managed-FFmpeg and H5J validation to platform packaging.

Earlier releases: [GitHub release history](https://github.com/sandorbx/MADI3D/releases).
