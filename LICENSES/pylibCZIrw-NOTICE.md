# pylibCZIrw and libCZI

MADI3D uses ZEISS `pylibCZIrw` 6.1.0, with a small reviewed source patch, to
read CZI files. The Python wrapper and the statically linked libCZI code in its
`_pylibCZIrw` extension are licensed under GNU LGPL version 3 or later. Copyright
notices in the exact source distribution identify Carl Zeiss Microscopy GmbH
(2022 for pylibCZIrw and 2017–2025 for libCZI). The upstream package notice
credits Carl Zeiss Microscopy GmbH. The extension also contains pybind11 under
its BSD-style license and other components whose notices are retained with the
matching source. `LGPL-3.0.txt` supplies the LGPL terms; the full libCZI LGPL
notice, bundled third-party notices, and pybind11 license are retained in the
adjacent `libCZI-LGPL-3.0.txt`, `libCZI-THIRD-PARTY-LICENSES.txt`, and
`pybind11-LICENSE.txt` files.

ZEISS is a registered trademark of Carl Zeiss AG. MADI3D is an independent
project and is not affiliated with, sponsored by, or endorsed by Carl Zeiss AG
or Carl Zeiss Microscopy GmbH.

The application uses the same pylibCZIrw API on Windows x64, Linux x64, macOS
Intel x64, and macOS Apple Silicon. The replacement is a matched set: the
replaceable Python files in `pylibCZIrw/` and the native `_pylibCZIrw` extension
must come from the same build and match the application's CPython ABI and
architecture. On Windows and Linux they are separate files beneath the
application's `_internal/` directory. In the macOS app bundle, Python files are
under `Contents/Resources/` and the extension is under `Contents/Frameworks/`.
Use `LICENSES/THIRD-PARTY-BUILD.json` in the exact package to find the recorded
paths and hashes; do not assume a filename suffix across platforms.

## Corresponding source and rebuild

The matching `MADI3D-<version>-third-party-sources-<platform>.zip` release asset
contains the exact `pylibczirw-6.1.0.tar.gz` source distribution. It includes
the libCZI and pybind11 source snapshots used by that release. The artifact also
contains the pinned zstd and Eigen3 source, MADI3D's reviewed patch, the build
recipe, constraints, source manifest, and notices. Verify its `CORRESPONDING-SOURCE.json`
hashes and platform/package association before rebuilding.

With CPython 3.11.9 and the matching platform toolchain installed, use the
included `scripts/build_czi.py` recipe and constraints. Put the exact manifest-
named source archives from the release asset in the recipe work directory's
`downloads/` folder; it verifies and reuses those cached files. If they are not
cached, it retrieves the pinned URLs and verifies the same hashes. The recipe
applies only the retained patch and checks source identities before compiling.
Install the resulting compatible wheel into a clean matching
Python environment, then replace both the `pylibCZIrw/` Python source tree and
the `_pylibCZIrw` extension at the paths identified by the package inventory.
Replacing just one half can pair incompatible APIs and binaries.

For a modified macOS app, sign the modified code and bundle with the user's own
identity or an ad-hoc signature before launching it; the original MADI3D
signature cannot remain valid after replacement. A signing key from MADI3D is
not required to replace or debug the library.

## LGPL rights

Recipients may modify and redistribute pylibCZIrw/libCZI under the LGPL,
replace the library with a compatible modified version, and reverse engineer
MADI3D as necessary to debug modifications to this library. MADI3D's terms do
not restrict those rights. The source archive and build material are provided
with the corresponding application release while that binary is distributed.
