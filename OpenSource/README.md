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

## Initial AGPL-3.0-only scope

The first public scientific-source release deliberately exposes the reusable
scientific/core implementation rather than the private application's GUI
panels. The exact project-owned Python files in scope are:

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

### Volume rendering

- `madi3d_app/volume/rendering.py`

The scope applies only to the exact project-owned files listed above. In
particular, private application GUI composition such as
`madi3d_app/registration/panel.py`, `madi3d_app/stitching/panel.py`, and the
general render-window/application infrastructure is not included unless it is
explicitly added to this manifest in a later public-source release.

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
