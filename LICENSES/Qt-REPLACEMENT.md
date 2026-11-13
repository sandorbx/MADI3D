# Replacing Qt, PySide6 and Shiboken6 in MADI3D

You may install and run MADI3D with compatible modified LGPL libraries, including
for debugging your changes. Keep a separate copy of the complete application
folder or bundle for this work. Close MADI3D before replacing files. Restore the
original copy if your replacement cannot load. MADI3D does not require a hash,
signature or license check of the original Qt libraries at application startup.

## Obtain or build compatible replacements

Use the same operating system, CPU architecture, Python ABI, Qt ABI, and release
configuration as your package. `LICENSES/Qt-BUILD.json` records the runtime
versions and platform; the release's platform `-environment.txt` records the
Python and packaging dependencies. The reference versions are Qt 6.9.3,
PySide6_Essentials 6.9.3 and shiboken6 6.9.3. A compatible modified version need
not retain the original version string.

Download the two source archives and `Qt-SOURCES.json` from the same MADI3D
release. Verify SHA-256 before extracting. The Qt archive contains its CMake
build system, `configure`/`configure.bat`, platform definitions and third-party
source. The pyside-setup archive contains `setup.py`, `build_scripts/`,
`coin_build_instructions.py`, `coin/`, `sources/pyside6/`, and
`sources/shiboken6/`. Preserve their license and copyright files.

For a source build, install the toolchain required by Qt 6.9 for your platform
(CMake/Ninja and MSVC on Windows, Xcode command-line tools on macOS, or the
supported GCC toolchain on Linux). Build Qt as **shared release libraries** for
the package's architecture. QtCore, QtGui, QtWidgets and QtSvg and their runtime
dependencies/plugins must remain available. Build PySide/Shiboken against that
Qt installation using the archived setup.py and its build requirements, for
example:

```text
python setup.py bdist_wheel --qtpaths=/path/to/modified/Qt/bin/qtpaths --standalone
```

Use `qtpaths.exe` on Windows. The archived `coin/` configurations and build
scripts describe upstream wheel production. The upstream build documentation
explains toolchain prerequisites and options:
https://doc.qt.io/archives/qt-6.9/build-sources.html and
https://doc.qt.io/qtforpython-6/building_from_source/index.html.
Online Qt for Python documentation follows the current release; the 6.9.3
archive's `sources/pyside6/doc/building_from_source/` documentation and build
scripts are authoritative for the source supplied here.
MADI3D consumes upstream wheels; it has no private Qt/PySide/Shiboken source
patches or private library-build scripts. MADI3D application source is not needed
to replace these dynamically loaded components.

## Windows x64 and Linux x64

Extract the ordinary MADI3D ZIP or TAR.GZ into a writable directory. The runtime
is in `MADI3D/_internal/`. Replace the relevant files in its `PySide6/` and
`shiboken6/` directories, keeping their relative layout. Include changed Python
`.py` files, extension modules (`.pyd` or `.so`), Qt and binding libraries
(`.dll` or `.so`), and any matching Qt plugins. On Linux, Qt libraries and plugins
are normally below `PySide6/Qt/lib/` and `PySide6/Qt/plugins/`; preserve symlinks,
permissions and library search paths. Remove stale `__pycache__` entries for
Python files you replace. Keep unrelated libraries and application files.

Run `MADI3D/MADI3D.exe` on Windows or `MADI3D/MADI3D` on Linux normally. Qt's
platform plugin must match the replacement Qt. For loader diagnostics, launch
from a terminal with `QT_DEBUG_PLUGINS=1`. Restore the complete original copy to
undo the replacement. The replacement is local; it does not change the system Qt.

## macOS Apple Silicon and Intel

Copy `MADI3D.app` to a writable location and choose **Show Package Contents**.
PyInstaller places native code under `Contents/Frameworks/` and data/Python files
under `Contents/Resources/`, with links between them. Replace the matching
`PySide6/` and `shiboken6/` files in those locations, following existing symlinks
to their targets. Preserve framework structure, symlinks and install names;
include corresponding plugins. Use arm64 libraries for Apple Silicon and x86_64
libraries for Intel. Remove stale `__pycache__` entries for replaced Python files.

Changing native code invalidates its existing ad-hoc signature. Sign your local
replacement bundle with your own ad-hoc signature (no developer certificate or
MADI3D key is needed), then verify it:

```sh
codesign --force --deep --sign - /path/to/MADI3D.app
codesign --verify --deep --strict /path/to/MADI3D.app
```

Open the modified app normally or run `Contents/MacOS/MADI3D` from Terminal to
see diagnostics. If macOS requests approval, use its normal **Open** or
**Privacy & Security** controls after inspecting your local build.

## Microsoft Store / MSIX

Keep the installed Store package intact. Windows protects installed package
files, so use a separate writable application copy for compatible replacements.
Close MADI3D, then run the following in PowerShell as the user who installed it:

```powershell
$package = Get-AppxPackage -Name Sandorbx.MADI3D
if ($null -eq $package) { throw "Install MADI3D from the Microsoft Store first." }
$replacement = Join-Path $env:LOCALAPPDATA ("MADI3D-Qt-replacement-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $replacement | Out-Null
Copy-Item -LiteralPath (Join-Path $package.InstallLocation "MADI3D.exe") -Destination $replacement
Copy-Item -LiteralPath (Join-Path $package.InstallLocation "_internal") -Destination $replacement -Recurse
Write-Output $replacement
```

Replace compatible components under `$replacement\_internal\PySide6` and
`$replacement\_internal\shiboken6` as described above, keeping the relative
layout and matching native plugins. Remove stale `__pycache__` entries for the
Python files you change. Run the modified copy normally:

```powershell
& (Join-Path $replacement "MADI3D.exe")
```

This runs the application as an ordinary desktop executable outside the Store
package. It requires neither a MADI3D signing key nor a new signing certificate.
Do not take ownership of WindowsApps, disable package protection, or change the
installed copy. The modified copy uses the application's ordinary local settings
and data locations; preserve your projects and preferences before experimenting.
To restore the original application, close the modified copy and launch MADI3D
from the Store/Start menu. The separate replacement folder can then be removed.
Store updates do not update this separate copy; create a fresh copy when needed.

On 8 October 2026 this procedure was tested without elevation on Windows 11 Pro
build 26200, with the Store-installed x64 package `2026.10.7.0` (MADI3D 0.33.03),
Python 3.11.9 and Qt/PySide/Shiboken 6.9.3.
A deliberate change to the copied PySide Python module executed. A separate
test changed the copied native QtCore library's version-string data: the modified
library loaded without re-signing, its marker was observed, and the normal MADI3D
main window rendered and exited successfully. This tested modified source-module
and native-library execution; it did not compile Qt from source.
