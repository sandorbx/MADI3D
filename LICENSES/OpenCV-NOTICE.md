# OpenCV and its linked codec libraries

MADI3D uses OpenCV headless 5.0.0.93 under Apache-2.0. The binding's LICENSE.txt
and full LICENSE-3RD-PARTY.txt from the exact build are copied into
`LICENSES/bundled/opencv-python-headless/`. These include codec, OpenSSL, image
library, numerical-library and other upstream terms. OpenCV's static internal
image-processing libraries are distinct from its FFmpeg video backend.

`NATIVE-WHEELS.json` records the hashes, native filenames and embedded FFmpeg
configuration/license evidence of the upstream wheels inspected on 2026-09-13.

* Windows x64: `opencv_videoio_ffmpeg500_64.dll` is a separate replaceable plugin.
  It contains static LGPL-2.1-or-later FFmpeg, libvpx, AOM, OpenCV core code,
  compiler-runtime code, and an OpenH264 dynamic-loader wrapper using API headers.
  The wrapper does not statically incorporate the OpenH264 codec implementation.
  Its configuration does not enable GPL/nonfree code. Replacing FFmpeg within
  this plugin requires rebuilding the **whole plugin**, not substituting an
  unrelated ffmpeg.exe. The upstream build recipe is in OpenCV's pinned
  `3rdparty/ffmpeg/ffmpeg.cmake` and the referenced `opencv_3rdparty` revision.
* Linux and macOS: MADI3D builds the same OpenCV version from pinned source,
  linked to separately built LGPL-only FFmpeg (8.1.1 on Linux, 8.1.2 on macOS).
  `scripts/build_macos_opencv.py` supports all three Unix targets. GPL/nonfree,
  external dependency autodetection and LAPACK are disabled. The Linux upstream
  wheel's OpenBLAS/libquadmath closure is not included. Actual source/configuration
  receipts and native/frozen video checks are required before distribution.

The upstream macOS arm64 wheel contains Homebrew FFmpeg 7.1.1 configured with
`--enable-gpl --enable-version3`, x264, x265, vidstab, rubberband and libpostproc.
It reports **GPL version 3 or later**. It must not be used in proprietary MADI3D
packages. This finding also occurs in the inspected 4.10, 4.11, 4.12 and 4.14
arm64 wheels; selecting a different version is not an established fix. The
5.0.0.93 Intel wheel has a different native dependency set and no separate libav
libraries; it is not evidence about the arm64 wheel.

## Replacement and source

Close MADI3D and copy its complete folder/bundle before replacing libraries.
On Windows/Linux, native files reside under `MADI3D/_internal/`, normally in
`cv2/` or `opencv_python_headless.libs/`. On macOS they are under
`MADI3D.app/Contents/Frameworks/cv2/`, including `.dylibs/`; resource symlinks are
maintained by PyInstaller. Use matching architecture and ABI. A custom FFmpeg
build must retain the backend's required symbols and codec functionality.

MADI3D permits modification, replacement, relinking and reverse engineering for
debugging these library modifications. No MADI3D signing key is required. On
macOS, follow the bundle re-signing procedure in `Qt-REPLACEMENT.md` after changing
native files. There is no startup hash restriction on these libraries.

The full LGPL-2.1 terms are in `LGPL-2.1.txt`. Source/build provenance and release
source obligations are tracked in `THIRD-PARTY-SOURCES.json`; distribution must
provide the corresponding sources, including scripts and any source changes,
alongside the application. Upstream links alone are not a claim that a MADI3D
release has provided all required source material.

The source-delivery tool creates a versioned per-platform archive tied to the
actual package/native hashes. Missing required library sources block publication;
optional forensic provenance mappings do not.
The matching Windows source branch is
`664c0098dcb47b361f20c1d6a518653c23f5f2b5`. Its empty OpenH264 header archive and
incomplete OpenCV wrapper snapshot are supplemented by full pinned source trees;
full GCC/MinGW compiler ancestry remains advisory. Eligible runtime combinations
use the GCC Runtime Library Exception; applicable notices are retained. See
`docs/third-party-source-delivery.md` in the development repository.

Upstream: https://github.com/opencv/opencv-python/tree/b83046cda41133f1bf2e73e99dba16a1248f103a
and https://ffmpeg.org/legal.html.

MADI3D's managed GPL FFmpeg executable is a different, separately downloaded
program. See `FFmpeg-NOTICE.md`; it does not describe the libav code above.
