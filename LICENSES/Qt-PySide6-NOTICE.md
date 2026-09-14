# Qt, PySide6 and Shiboken6

MADI3D uses Qt 6.9.3, PySide6 6.9.3 (PySide6_Essentials), and Shiboken6 6.9.3
under the GNU Lesser General Public License version 3 (LGPLv3).
Copyright (C) The Qt Company Ltd. and other contributors. The original
file-specific copyright and license notices remain in the corresponding source
archives and the packaged distribution metadata.

MADI3D uses the upstream binary wheels without source modifications to these
libraries. PyInstaller arranges their files, adjusts native library load paths
where needed, and signs the macOS bundle. Those packaging operations are not
source patches. Qt's own third-party components retain their respective licenses;
the full Qt source archive includes their notices. The source archive also
contains modules that MADI3D does not ship; their presence is not a grant to use
GPL-only modules in a proprietary application.

The complete LGPLv3 and the GPLv3 terms it incorporates accompany this notice as
`LGPL-3.0.txt` and `GPL-3.0.txt`. In MADI3D, open **License** to read the Qt notice,
license texts, source information and replacement instructions.

## Your rights

You may study, modify and redistribute these libraries under their licenses,
replace them with compatible modified versions, and run MADI3D with the replaced
libraries. You may reverse engineer MADI3D as necessary to debug such library
modifications. MADI3D's proprietary restrictions do not limit these rights.
See `Qt-REPLACEMENT.md` for installation and replacement instructions for
Windows, Linux and both macOS architectures. No MADI3D signing key is required.

## Corresponding source, provided by MADI3D

For each public binary release carrying this notice, download the following
assets from **the same MADI3D release** at
https://github.com/sandorbx/MADI3D/releases (select the MADI3D version shown by
the application):

- `qt-everywhere-src-6.9.3.tar.xz`: complete upstream Qt source, including
  configuration/build scripts and Qt's third-party sources and notices.
- `pyside-setup-everywhere-src-6.9.3.tar.xz`: complete upstream Qt for Python
  source, including **both PySide6 and Shiboken6**, generators and build scripts.
- `Qt-SOURCES.json`: exact versions, source filenames, sizes, SHA-256 hashes,
  and upstream provenance.
- `Qt-REPLACEMENT.md` and the platform's `-environment.txt`: build/replacement
  instructions and the resolved MADI3D packaging environment.
- `SHA256SUMS.txt`: checksums for the binary and corresponding source assets.

These are free downloads from the MADI3D-controlled public repository, alongside
the binaries, under GPLv3 section 6(d) as incorporated by LGPLv3. They require no
purchase or source request. The upstream URLs in `Qt-SOURCES.json` record origin;
MADI3D provides its own copies as release assets and retains them while the
corresponding binaries are offered. GitHub's automatically generated MADI3D
"Source code" archives are not the Qt/PySide/Shiboken source archives.

The packaged `Qt-BUILD.json` identifies this build's Qt/PySide/Shiboken versions,
Python, architecture, and release URL. Development and diagnostic builds may
precede publication; do not redistribute them without also providing the source
assets described here. If a public release's source asset is unavailable, report
the release version and filename to developer@madi3d.org.
