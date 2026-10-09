# MADI3D changelog

## 0.33.04 — 2026-10-09

- MADI3D is available free for Windows x64 from the [Microsoft Store](https://apps.microsoft.com/detail/9NX4SRPXVG23). Store and GitHub updates may become available at different times.
- Fixed SVG icon initialization during Windows Store startup and added Help → Privacy Policy.
- Improved project dirty tracking, unsaved-data indicators, scene-tree search and scope selection, project replacement confirmation, and mesh-queue progress. Successful task completion no longer triggers cancellation status.
- Improved stability of background volume, mesh and media loading and project cleanup. NeuronBridge source-list refresh handles removed sources correctly.
- Added readable volume-history summaries while retaining scientific history as the authoritative record and accepting supported legacy result shapes.
- Improved registration/reformat output identity, Windows output-path handling and uncertain-unit geometry provenance; refined animation timelines, playback, camera zoom and transform controls.

## 0.33.0 — 2026-09-30

### Added

- Read-only ZEISS CZI and Nikon ND2 import with explicit acquisition/channel selection, supported time-series handling and rejection of ambiguous layouts.
- PLY mesh import/export and project resources.
- Animation storyboard editing, object-motion and clipping capture, recorded-take trimming, preview and tiled export.

### Improved

- Stitching applies successful alignment to the scene, exposes apply/restore controls and retains the selected registration model during its automatic-placement retry. Cancellation and failed alignment leave tile poses intact.
- Stitched output names preserve anchor/channel labels; registration reformats preserve scene groups, geometry and generated-volume provenance with atomic output handling.
- Project save/reload, missing-source relinking and bounded recovery of supported historical packages preserve usable state without inventing missing evidence.
- Physical-grid calibration retains acquisition-wide correction history and marks affected completed results out of date.
- Automatic NeuronBridge MIPs and queries remain available for preview/search after project reload; result selections and scene associations preserve identity.
- Transform-tool tabs, scene-tree context menus, timeline interaction and selection-based animation actions. Removed the source-replacement control from the transform tools.

### Compatibility and scope

- Linux requires Ubuntu 20.04 or newer x64 / glibc 2.31 or newer.
- CZI color/BGR data and unsafe or ambiguous layouts remain unsupported. Local Color-Depth scores do not establish equivalence to hosted scoring or biological accuracy; hosted custom-query submission remains unavailable.

## 0.32.0 — 2026-09-13

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

### Licensing and scope

- Qt/PySide6/Shiboken6 LGPL notices, complete license texts, explicit replacement/debugging rights, and source/build instructions accompany the packages. Exact corresponding source archives are available with each release.
- Local positive-CDS scores are distinct from hosted scores; authenticated hosted custom-query submission is not included. Full-library performance and biological retrieval accuracy are not established. SWC queries are rendered-surface projections, not canonical skeleton queries.

## 0.31.3 — 2026-09-11

- Segmentation extraction preserves the complete source grid for selected voxels and original signal. Outside-mask voxels are zero; original-signal extraction retains its configured mask margin.
- Updated extraction guidance and microscopy import/decode documentation.

## 0.31.2 — 2026-09-10

- Managed FFmpeg downloads use operating-system HTTPS trust, with pinned release and checksum validation.
- Added managed native CMTK installation for Apple Silicon, with cancellation, rollback and retained backend selection.
- Corrected CAVE color-update pacing and connection lifecycle handling.

Earlier releases: [GitHub release history](https://github.com/sandorbx/MADI3D/releases).
