# MADI3D Open-Source Scientific Components

This directory defines the open-source licensing boundary for selected
project-owned scientific source in MADI3D.

MADI3D keeps one canonical implementation of each component. Source is not
copied into a parallel tree merely to create a second license location. The
project-owned source listed below remains at its normal package path and is
licensed to recipients under the GNU Affero General Public License version 3
only (`AGPL-3.0-only`) when distributed or made available by the MADI3D
project.

`OpenSource/LICENSE` contains the applicable license text.

## Current AGPL-3.0-only scope

The public scientific-source scope deliberately exposes reusable scientific
and data-integration implementation rather than the private application's GUI
composition. The exact project-owned Python files in scope are:

### Registration core

- `madi3d_app/registration/__init__.py`
- `madi3d_app/registration/models.py`
- `madi3d_app/registration/output.py`
- `madi3d_app/registration/service.py`

### CMTK integration used by registration

- `madi3d_app/integrations/cmtk/__init__.py`
- `madi3d_app/integrations/cmtk/backend.py`
- `madi3d_app/integrations/cmtk/process.py`
- `madi3d_app/integrations/cmtk/registration.py`
- `madi3d_app/integrations/cmtk/setup.py`
- `madi3d_app/integrations/cmtk/xform.py`

### 3D microscopy stitching core

- `madi3d_app/stitching/__init__.py`
- `madi3d_app/stitching/models.py`
- `madi3d_app/stitching/service.py`
- `madi3d_app/stitching/stitching_positioning.py`
- `madi3d_app/stitching/workers.py`

### Volumetric segmentation

- `madi3d_app/segmentation/__init__.py`
- `madi3d_app/segmentation/controller.py`

### NeuronBridge and Color-Depth MIP integration

- `madi3d_app/integrations/neuronbridge/__init__.py`
- `madi3d_app/integrations/neuronbridge/assets.py`
- `madi3d_app/integrations/neuronbridge/cache.py`
- `madi3d_app/integrations/neuronbridge/cave_pacing.py`
- `madi3d_app/integrations/neuronbridge/cdm.py`
- `madi3d_app/integrations/neuronbridge/cdm_export.py`
- `madi3d_app/integrations/neuronbridge/cdm_lut.py`
- `madi3d_app/integrations/neuronbridge/cdm_records.py`
- `madi3d_app/integrations/neuronbridge/comments.py`
- `madi3d_app/integrations/neuronbridge/csv_parser.py`
- `madi3d_app/integrations/neuronbridge/download_worker.py`
- `madi3d_app/integrations/neuronbridge/evidence.py`
- `madi3d_app/integrations/neuronbridge/geometry_input.py`
- `madi3d_app/integrations/neuronbridge/import_service.py`
- `madi3d_app/integrations/neuronbridge/local_catalog.py`
- `madi3d_app/integrations/neuronbridge/local_library.py`
- `madi3d_app/integrations/neuronbridge/local_scorer.py`
- `madi3d_app/integrations/neuronbridge/local_search.py`
- `madi3d_app/integrations/neuronbridge/local_smoke.py`
- `madi3d_app/integrations/neuronbridge/mapping.py`
- `madi3d_app/integrations/neuronbridge/message_client.py`
- `madi3d_app/integrations/neuronbridge/public_api.py`
- `madi3d_app/integrations/neuronbridge/public_cache.py`
- `madi3d_app/integrations/neuronbridge/query.py`
- `madi3d_app/integrations/neuronbridge/records.py`
- `madi3d_app/integrations/neuronbridge/remote.py`
- `madi3d_app/integrations/neuronbridge/search_profiles.py`
- `madi3d_app/integrations/neuronbridge/summary.py`
- `madi3d_app/integrations/neuronbridge/thumbnails.py`
- `madi3d_app/integrations/neuronbridge/workspace.py`

This scope covers the reusable query/evidence records, Color-Depth MIP
generation and export, local-library search, public-result lookup, asset/cache
handling, and NeuronBridge data-service integration. Private GUI composition
and application controllers remain outside this scope unless explicitly listed
in a later public-source release.

### Shared dependencies used by the public scientific components

- `madi3d_storage.py`
- `madi3d_app/volume/resampling.py`

The storage helper supplies the canonical cross-platform cache/data locations
used by the NeuronBridge cache and local-library code. The Qt/VTK-free affine
sampler is shared by Color-Depth MIP generation and registration. Publishing
these dependencies keeps the exposed scientific source directly traceable
without duplicating or rewriting their implementation.

### Microscopy volume reading and source interpretation

- `madi3d_app/io_utils.py`
- `madi3d_app/integrations/ffmpeg/__init__.py`
- `madi3d_app/integrations/ffmpeg/backend.py`
- `madi3d_app/volume/decode.py`
- `madi3d_app/volume/geometry.py`
- `madi3d_app/volume/leica_lif.py`
- `madi3d_app/volume/microscopy_metadata.py`
- `madi3d_app/volume/olympus.py`
- `madi3d_app/volume/probe.py`
- `madi3d_app/volume/provenance.py`
- `madi3d_app/volume/source_formats.py`
- `madi3d_app/volume/zeiss_lsm.py`

This scope makes the project-owned file/container interpretation, source and
channel metadata, calibration evidence, physical/working-grid handling, and
voxel decoding logic auditable. The GUI-independent FFmpeg backend used by H5J
decoding is included; its Qt first-use setup UI remains private application
composition. The scope does not include private GUI import composition, scene
publication, or other application orchestration that is not listed above.

### Volume rendering

- `madi3d_app/volume/rendering.py`

The scope applies only to the exact project-owned files listed above. In
particular, private application GUI composition such as
`madi3d_app/registration/panel.py`, `madi3d_app/stitching/panel.py`, the
NeuronBridge UI/controllers, and the general render-window/application
infrastructure is not included unless it is explicitly added to this manifest
in a later public-source release.

Third-party software, binaries, data, templates, icons, codecs, and other
material retain their own licenses and notices.

Files outside the exact scope above remain governed by the root MADI3D
`LICENSE` unless an explicit file-specific or third-party license says
otherwise.

## Dual licensing

The AGPL grant applies to recipients of the open-source scientific components.
It permits use, modification, redistribution, and commercial activity subject
to AGPL-3.0-only.

Official MADI3D releases may use the same source under separate rights held by
or granted to the MADI3D project. That separate project license is what permits
the official proprietary MADI3D application to incorporate these components
without changing the license of unrelated proprietary MADI3D source.

Contributions to this scope therefore need rights compatible with both public
AGPL-3.0-only publication and official MADI3D's separate project license. See
`CONTRIBUTOR_AGREEMENT.md`.

## Publication

This directory is a license and publication manifest; it is not a duplicate
source tree.

A private MADI3D development checkout is not itself a public release. When the
scientific components are published, the publication must include the
applicable canonical source, this scope notice, and the AGPL license information
required by AGPL-3.0-only.

The public mirror is intentionally maintained from an explicit file allowlist.
New files are not published merely because they are added under a neighboring
private package directory; expanding the public scope requires an explicit
manifest and workflow change.
