# MADI3D changelog

## 0.36.22 — Public beta

- **Guided workflows:** New Registration, 3D Stitching and Color-Depth MIP (CDM) Search wizards, available from the **Wizards** toolbar menu.
- **Registration and stitching:** Groupwise registration is available in the Registration workspace. Registration and stitching load the data they need as they run, and stitched results preserve all channels.
- **Neuron search:** Preview a CDM from microscopy data, draw a search area, and search an installed library. FlyWire results are easier to review and add to the scene.
- **Progress and scene controls:** Follow background work in one Activity Center. The scene tree offers clearer visibility controls and options to show or sort checked objects.
- **Segmentation and projects:** More responsive brush previews, restored segmentation when reopening projects, and clearer messages when saving encounters a problem. Undo/Redo history remains available only during the current session.
- **Files and measurements:** Improved handling of large NRRD files and OBJ files with textures, plus corrected calibrated mesh area and volume measurements.

## 0.34.0 — 2026-10-20

- Registration and stitching can work with volumes that are not already loaded into the viewer.
- Stitching saves every output channel and can leave large results unloaded until needed.
- Transforms stay consistent across channels, including unloaded ones.
- Segmentation work is restored when reopening a saved project. Undo/Redo history remains session-only.

## 0.33.04 — 2026-10-09

- MADI3D became available for Windows through the [Microsoft Store](https://apps.microsoft.com/detail/9NX4SRPXVG23).
- Improved scene-tree search, selection and unsaved-change indicators.
- More reliable background loading and project cleanup.
- Improved registration output handling, animation playback, camera controls and transforms.

## 0.33.0 — 2026-09-30

- Added ZEISS CZI and Nikon ND2 microscopy import, including supported multichannel and time-series data.
- Added PLY mesh import and export.
- Expanded animation editing with object movement, clipping, preview, trimming and video export.
- Improved automatic stitching alignment and handling of cancelled or failed alignment.
- Improved project recovery, source-file relinking and physical calibration.
- Saved Color-Depth MIPs and NeuronBridge search results can be reopened for further comparison.

## 0.32.0 — 2026-09-13

- Added Color-Depth MIP generation and local neuron searches using installed brain and VNC libraries.
- Added NeuronBridge result browsing, selective neuron loading and saved searches.
- Added automatic overlap detection for 3D stitching.
- Improved segmentation responsiveness, camera controls and handling of large image files.

## 0.31.3 — 2026-09-11

- Improved export of segmented volumes, preserving the original image grid and selected signal.

## 0.31.2 — 2026-09-10

- Improved automatic downloads of video-processing tools.
- Added guided CMTK installation for Apple Silicon Macs.
- Improved the reliability of CAVE connections.

Earlier releases: [GitHub release history](https://github.com/sandorbx/MADI3D/releases).
