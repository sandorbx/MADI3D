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
and tied to each final package's SHA-256 and native inventory.

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
| imageio, tifffile, liffile, oiffile, nibabel, pynrrd, nd2 | Bundled Python readers/writers; respectively BSD-2-Clause, BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, MIT, MIT and BSD-3-Clause. Nikon ND2 is read-only. No Bio-Formats/Java runtime is included. |
| pylibCZIrw / libCZI | Bundled, replaceable Python reader source and native extension, version 6.1.0 plus the reviewed CZI subblock-inspection patch; LGPL-3.0-or-later. The static libCZI implementation is part of `_pylibCZIrw`, not a separate runtime library. `libCZI-LGPL-3.0.txt`, `libCZI-THIRD-PARTY-LICENSES.txt`, and `pybind11-LICENSE.txt` retain its upstream licenses/notices. Matching source, build instructions, replacement locations, macOS re-signing, and LGPL rights are in `pylibCZIrw-NOTICE.md`. MADI3D uses it only for read-only grayscale fluorescence microscopy scanned-volume import. |
| xmltodict, validators | Python dependencies of pylibCZIrw; MIT, bundled when collected. |
| Eigen3 | Header-only Eigen3 3.4.0 is compiled into libCZI; MPL-2.0. The exact source revision and its upstream license are retained with the pylibCZIrw corresponding-source materials. |
| Dask and ND2 runtime dependencies | Bundled when collected: Dask, click, cloudpickle, fsspec, importlib-metadata, locket, ome-types, partd, pydantic-extra-types, PyYAML, resource-backed-dask-array, toolz, xsdata and zipp. Exact versions and licenses appear in `THIRD-PARTY.json` and the generated build inventory. |
| CloudVolume 12.15.2 | Bundled Python geometry decoder for FlyWire m783 mesh and skeleton retrieval; BSD-3-Clause. Its exact `LICENSE` and `AUTHORS` files are copied from the collected wheel. Other dependency notices are selected from the actual PyInstaller Analysis. |
| FlyWire decoder dependencies | Exact Analysis-selected Python/runtime components are pinned in `constraints-build.txt`; upstream notices and matching source archives are retained. `fastremap`, `crc32c`, `microviewer`, and `orjson` Python source is kept outside the executable archive so recipients can replace it while preserving the platform-specific extension layout. Native wheels are SHA-reviewed per target; package release checks still require the actual selected inventory and retained notices. |
| OpenCV headless | Bundled Python/native video and image operations; Apache-2.0 plus native notices and LGPL FFmpeg where present. See `OpenCV-NOTICE.md`. The upstream 5.0.0.93 macOS arm64 wheel is explicitly rejected. |
| pygame / SDL | Bundled replaceable pygame Python/native modules (LGPL-2.1-or-later) and SDL2 (zlib), with platform-specific native dependencies. See `pygame-NOTICE.md` and `NATIVE-WHEELS.json`. Unused GPL FreeSans and sample icon assets are excluded. |
| pypdfium2 / PDFium | Bundled PDF binding and renderer; BSD-3-Clause/Apache-2.0 and the wheel's complete PDFium build notices. See the existing `PDFium-NOTICE.md`. |
| psutil, threadpoolctl | Bundled Python/native runtime helpers; BSD-3-Clause. |
| scikit-image 0.26.0, lazy-loader 0.6 | Scalar slice-wise/3D CLAHE through `skimage.exposure`; scikit-image retains BSD-3-Clause, BSD-2-Clause and MIT notices, including Karel Zuiderveld's adaptive histogram equalization attribution. lazy-loader is BSD-3-Clause. Original wheel licenses are copied from the collected distributions into `LICENSES/bundled/`. No Fiji numerical equivalence is claimed. |
| pydantic, pydantic-core, annotated-types | Bundled validation libraries, including the native Rust extension; MIT. |
| requests, urllib3, truststore | Bundled HTTPS libraries; Apache-2.0, MIT and MIT. requests' NOTICE is retained. |
| pyarrow 26.0.0 | MCNS Feather annotation reader; Apache-2.0 with bundled third-party components. The wheel's complete LICENSE.txt and NOTICE.txt are retained by metadata collection. No pandas is required. |
| certifi, tqdm | Bundled in source form; MPL-2.0 (certifi), MPL-2.0/MIT (tqdm). `MPL-2.0.txt` supplies the full terms. |
| NeuronBridge Python client | Protocol reference for the native metadata adapter at commit `15a268c68a33983cbff6243fd99aa2efbed371ab`. The upstream package is no longer bundled. Its BSD-3-Clause notice is retained in `neuronbridge-python-LICENSE.txt` (HHMI copyright 2022). |
| Matplotlib and runtime helpers | Collected transitively through VTK/scientific libraries: Matplotlib, contourpy, cycler, fontTools, kiwisolver, packaging, pyparsing, dateutil, six, typing-extensions, importlib-resources, charset-normalizer, idna, colorama and setuptools runtime support. Exact selection and licenses appear in `THIRD-PARTY-BUILD.json`; DejaVu/STIX font notices are retained. |
| PyInstaller | Build tool; its bootloader and runtime hooks are also shipped. GPL-2.0-or-later **with the bootloader/distribution exception** permits packaging proprietary applications. Its complete COPYING/exception accompanies the application. |
| altgraph, pefile, pywin32-ctypes, unused PyInstaller hooks | Build-only tooling; not application dependencies. The build rejects their unexpected incorporation. Only the selected `pyi_rth_cryptography_openssl` runtime hook is bundled from pyinstaller-hooks-contrib; it is Apache-2.0. |
| Tabler icons | Bundled SVG assets; MIT, notices in `Tabler-Icons-LICENSE.txt` and `Tabler-Icons-NOTICE.md`. |
| NeuronBridge logo | HHMI BSD-3-Clause. Source notice in `docs/licenses/neuronbridge-logo.txt`; packaged as `LICENSES/neuronbridge-logo.txt`. |
| Positive CDS scorer and JavaScript oracle | Adapted scorer and retained `mipsearch.mjs` reference from JaneliaSciComp/neuronbridge-services `3847c602fcc5e3ae2bae47b7308ee11263ae6cef`. Copyright 2020 Howard Hughes Medical Institute, BSD-3-Clause. See `neuronbridge-services.txt`. |
| Embedded Color-Depth MIP palette | [PsychedelicRainBow2.lut](https://github.com/JaneliaSciComp/neuronbridge-aligners/blob/ba883d845edba801f2560f8aa9b772c4caec1f76/brain-aligner/aligner/fiji_luts/PsychedelicRainBow2.lut) from JaneliaSciComp/neuronbridge-aligners at `ba883d845edba801f2560f8aa9b772c4caec1f76`. Copyright 2022 Howard Hughes Medical Institute, BSD-3-Clause. See `neuronbridge-aligners-LICENSE.txt`. |
| Native NeuronBridge shape scorer and LM derivative producer | Adapted from JaneliaSciComp/EM_MIP_search `83ee6f73cee0708f7b013c66d9dbbe11a1b3e7b2`. Copyright 2023 Howard Hughes Medical Institute, BSD-3-Clause. See `EM_MIP_search-LICENSE.txt`. The independent producer oracle uses ImageJ 1.53v with the NIH public-domain disclaimer in `ImageJ-LICENSE.txt`. |
| Earlier gradient reference fixtures | JaneliaSciComp/colormipsearch `8b85c458f3b9231ec7bff5f1b4262cbf2abf8c3d`. Copyright 2020 Howard Hughes Medical Institute, BSD-3-Clause. See the [upstream notice](colormipsearch-LICENSE.txt) and the [public corpus guide](../OpenSource/neuronbridge/README.md). |
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
`archive/documentation/validation/third-party-distribution-audit.md` in the development repository. A passing
source-policy test alone is not four-platform binary verification.
