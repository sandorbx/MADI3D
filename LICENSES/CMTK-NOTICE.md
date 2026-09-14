# CMTK external toolkit

CMTK (Computational Morphometry Toolkit) is separate software licensed under the
GNU General Public License version 3. The toolkit retains its contributors'
copyright and component-specific notices; see its `core/LICENSE`,
`core/Licenses/` and the GPLv3 text supplied here as `GPL-3.0.txt`.
Upstream project: https://www.nitrc.org/projects/cmtk/.

MADI3D does not bundle CMTK executables, shared libraries, installers or a WSL
image. Its integration invokes external programs as subprocesses, passing files
and command-line arguments. Installation occurs only after user action.

| Platform | How CMTK is obtained |
| --- | --- |
| Windows x64 | The user may select an existing WSL installation. Managed setup downloads a checksum-verified Ubuntu 24.04.4 LTS WSL image into application data, creates the isolated MADI3D distribution and installs CMTK through Ubuntu apt. The Ubuntu package version is repository-selected and recorded by runtime probing; it is not a pinned 3.4.0 binary. |
| Ubuntu/Linux x64 | An existing native installation, or explicit apt installation on supported Ubuntu/Debian hosts. The distribution supplies the package, its copyright files and source package. |
| macOS Apple Silicon | Explicit download of `cmtk-3.4.0-dev-macos-arm64-gcd.tar.gz` from jefferis/cmtk `natdev-latest`, SHA-256 `4160852416b6cba9a55c7e7d0497e4d7153262a3a5265cafb39d1bb850649d6c`, extracted under the user's application data. |
| macOS Intel | User-provided compatible native installation or source build. Managed download is not offered for Intel; the arm64 archive is not installed under emulation. |

The macOS asset checksum was rechecked against the provider's release metadata on
2026-09-13. The corresponding release target is
`bfc0c235099e4499b76f1f7977aee73933ab7d7e`:
https://github.com/jefferis/cmtk/tree/bfc0c235099e4499b76f1f7977aee73933ab7d7e.
The rolling tag can change; a replacement asset fails checksum verification until
reviewed. Source/build provenance is recorded separately from that mutable tag.

Release: https://github.com/jefferis/cmtk/releases/tag/natdev-latest.
For Ubuntu packages, retain `/usr/share/doc/cmtk/copyright` and obtain the source
matching the installed package using the distribution's source repositories.
MADI3D does not redistribute those packages or promise a fixed apt version.

On macOS MADI3D creates a small launcher in the external installation to handle
paths containing spaces. It does not patch the toolkit binaries or place them in
the MADI3D bundle. Download staging, cancellation cleanup and managed installation
directories remain outside distributable application resources.
