# Third-party software distributed with MADI3D

MADI3D's licensor and developer is **Sandor Kovacs** (developer@madi3d.org;
https://madi3d.org/impressum/). MADI3D's own terms are in `../LICENSE`.
Third-party components retain their own licenses, including the rights to
modify, replace and debug LGPL libraries. Contributor credits are not changed
by this inventory.

`THIRD-PARTY.json` is the reviewed dependency policy. In each built application,
`THIRD-PARTY-BUILD.json` identifies the distributions and native files actually
collected by PyInstaller. Its `input_sha256` values identify files **before**
PyInstaller changes load paths or signs them. CI's separate distribution report
records hashes of the final native files. Neither file is a runtime integrity
restriction on replacement libraries.

Corresponding-source artifacts are generated from `THIRD-PARTY-SOURCES.json`
and tied to each final package's SHA-256 and native inventory. Release readiness
requires verification of archive contents, exact binary/source associations,
recipes, patches and notices. Deleting a blocker record does not supply missing
evidence. Incomplete diagnostic source artifacts remain explicitly blocked.

The build copies original copyright, license, NOTICE and patent files from
**collected distributions only** into `LICENSES/bundled/`. This includes native
wheel notices, not just a package's metadata license label. Additional upstream
notices absent from wheels are retained here, with exact source paths and hashes
in `UPSTREAM-NOTICES.json`. Some upstream notice collections describe optional
code; the native inventory, rather than the presence of a license text, determines
what is actually shipped.

| Component | Distribution and license |
| --- | --- |
| CPython and standard library | Bundled interpreter, Python modules and native extensions; PSF/Python terms and bundled-library notices. `CPython-LICENSE.txt` is copied from the build interpreter. |
| PySide6 Essentials, Shiboken6, Qt | Bundled replaceable Python files and shared libraries, version 6.9.3; LGPL-3.0. See `Qt-PySide6-NOTICE.md`, `Qt-REPLACEMENT.md`, `Qt-SOURCES.json` and `Qt-native-NOTICES.txt`. Addons and modules outside the reviewed widget stack are rejected. |
| Qt's Windows software renderer | Bundled `opengl32sw.dll`, Mesa 11.2.2 / LLVM 3.6.2; MIT, University of Illinois/NCSA and retained permissive component terms. See `Mesa-LLVM-NOTICE.txt`. This has separate licensing from Qt. |
| VTK 9.6.2 | Python extensions/shared native libraries; BSD-3-Clause, with separately licensed vendored code in `VTK-native-NOTICES.txt`. Includes Viskores, CGNS, Exodus, HDF5, netCDF, FreeType, GL2PS, libharu, XML, image codecs and compression libraries. GL2PS is used under its permissive GL2PS alternative, not its LGPL alternative. |
| NumPy, SciPy | Python/native libraries; BSD-3-Clause plus wheel notices for BLAS/LAPACK, compiler runtimes, Highway, pocketfft, Qhull and other vendored code. GCC runtime exceptions must be retained where applicable. |
| SimpleITK / ITK | Bundled native extension; Apache-2.0 plus ITK's separately licensed third-party code. Original SimpleITK LICENSE/NOTICE and `ITK-native-NOTICES.txt` accompany the package. |
| Pillow | Bundled Python/native image library; MIT-CMU and codec/font library notices in the wheel's full license file. |
| h5py / HDF5 | Bundled Python extensions/native HDF5; BSD-3-Clause and the HDF5, LZF and other notices shipped by the exact wheel. |
| imagecodecs | Bundled selected native codec extensions; BSD-3-Clause and individual codec notices. A license for an optional HEIF or Jetraw backend does not mean that backend is bundled. Unresolved external codec DLLs must not be added implicitly. |
| imageio, tifffile, liffile, oiffile, nibabel, pynrrd | Bundled Python readers/writers; respectively BSD-2-Clause, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, MIT and MIT. No Bio-Formats/Java runtime is included. |
| OpenCV headless | Bundled Python/native video and image operations; Apache-2.0 plus native notices and LGPL FFmpeg where present. See `OpenCV-NOTICE.md`. The upstream 5.0.0.93 macOS arm64 wheel is explicitly rejected. |
| pygame / SDL | Bundled replaceable pygame Python/native modules (LGPL-2.1-or-later) and SDL2 (zlib), with platform-specific native dependencies. See `pygame-NOTICE.md` and `NATIVE-WHEELS.json`. Unused GPL FreeSans and sample icon assets are excluded. |
| pypdfium2 / PDFium | Bundled PDF binding and renderer; BSD-3-Clause/Apache-2.0 and the wheel's complete PDFium build notices. See the existing `PDFium-NOTICE.md`. |
| psutil, threadpoolctl | Bundled Python/native runtime helpers; BSD-3-Clause. |
| pydantic, pydantic-core, annotated-types | Bundled validation libraries, including the native Rust extension; MIT. |
| requests, urllib3, truststore | Bundled HTTPS libraries; Apache-2.0, MIT and MIT. requests' NOTICE is retained. |
| certifi, tqdm | Bundled in source form; MPL-2.0 (certifi), MPL-2.0/MIT (tqdm). `MPL-2.0.txt` supplies the full terms. |
| NeuronBridge Python client | Bundled `neuronbridge.client` and models at commit `15a268c68a33983cbff6243fd99aa2efbed371ab`; BSD-3-Clause, HHMI copyright 2022. See `neuronbridge-python-LICENSE.txt`. Its optional distributed validation/profiling tools are not bundled. |
| Matplotlib and runtime helpers | Collected transitively through VTK/scientific libraries: Matplotlib, contourpy, cycler, fontTools, kiwisolver, packaging, pyparsing, dateutil, six, typing-extensions, importlib-resources, charset-normalizer, idna, colorama and setuptools runtime support. Exact selection and licenses appear in `THIRD-PARTY-BUILD.json`; DejaVu/STIX font notices are retained. |
| PyInstaller | Build tool; its bootloader and runtime hooks are also shipped. GPL-2.0-or-later **with the bootloader/distribution exception** permits packaging proprietary applications. Its complete COPYING/exception accompanies the application. |
| altgraph, pefile, pywin32-ctypes, PyInstaller hooks | Build-only tooling; not application dependencies. The build rejects their unexpected incorporation. |
| Tabler icons | Bundled SVG assets; MIT, notices in `Tabler-Icons-LICENSE.txt` and `Tabler-Icons-NOTICE.md`. |
| NeuronBridge logo and local search attribution | Bundled asset/adapted algorithm notices in `../licenses/neuronbridge-logo.txt` and `../licenses/neuronbridge-services.txt`; HHMI BSD-3-Clause. |
| Managed FFmpeg | Separately downloaded GPLv3 executable, outside the application, invoked by subprocess. See `FFmpeg-NOTICE.md`. It is distinct from OpenCV's linked libav libraries. |
| CMTK | External GPLv3 toolkit; installed after user action or supplied by the user. Platform behavior is documented in `CMTK-NOTICE.md`. No CMTK executable or WSL image is an application resource. |
| NeuronBridge services, libraries, templates and datasets | Runtime network sources/user downloads, not frozen application software. Dataset-specific identity, provenance and permissions remain separate from the client software license. |
| PyMuPDF/MuPDF, PyMeshFix/MeshFix | Not shipped. Both Python and native reintroduction are rejected. |

Microsoft redistributable runtime DLLs and OS libraries are identified separately
in the generated inventory. Native files from Conda packages retain the notices
from their exact package records; Linux system libraries retain the owning Debian
package's copyright file. Unknown owners stop the build. Development environments
are not release evidence.

The project-owned scientific components listed in `../OpenSource/README.md` have
a separate dual-licensing arrangement described in `../LICENSE`. Their optional
AGPL source grant is not a third-party AGPL dependency.

Release audit evidence and outstanding platform/source verification are in
`docs/third-party-distribution-audit.md` in the development repository. A passing
source-policy test alone is not four-platform binary verification.
