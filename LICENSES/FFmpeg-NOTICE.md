# FFmpeg notice for MADI3D

MADI3D does not bundle an FFmpeg executable. H5J import/export requires an
external FFmpeg executable with HEVC decoding and libx265 encoding support.

MADI3D uses its verified managed binary for H5J and does not discover FFmpeg on
the system PATH. If the managed binary is unavailable, the user may explicitly
choose to download the pinned managed binary. MADI3D stores it in the current
user's application-data directory, verifies it against a pinned SHA-256 checksum
before use, and does not add it to the system PATH. Installation downloads a
pinned ZIP or TAR.XZ archive, verifies its SHA-256 checksum, streams only the pinned
executable member to a temporary file, verifies that file's SHA-256 checksum,
and then atomically installs it.

## Managed FFmpeg build

Provider/build projects: BtbN `FFmpeg-Builds` for Windows and Linux, and Martin
Riedl's FFmpeg build service for macOS.

Pinned release identity: `9.0.2-h5j12-2`

FFmpeg version: `9.0.2`

Build/release sources:
https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-09-30-13-08
https://ffmpeg.martin-riedl.de/

Windows and Linux use the retained September monthly build. BtbN retains monthly
builds for two years; daily-build URLs expire after fourteen builds. The macOS
pins are unchanged. Both archive and executable checksums remain mandatory.

The selected builds include x265 8-, 10- and 12-bit support and are licensed
under GPL version 3 or later. Their platform-specific ZIP or TAR.XZ archive and
extracted executable SHA-256 values are pinned in MADI3D's FFmpeg backend. The
Windows and Linux builds are GPL-enabled with nonfree components disabled.

FFmpeg project and source:
https://ffmpeg.org/
https://ffmpeg.org/download.html

x265 project and source:
https://bitbucket.org/multicoreware/x265_git/

The verified archive and executable identities are pinned in MADI3D's FFmpeg
backend. MADI3D does not include these executables in its installer.

The GNU General Public License version 3 text distributed with MADI3D is in:
`LICENSES/ffmpeg-gplv3.txt`.

FFmpeg/x265 remain separate external software. MADI3D invokes the selected
FFmpeg executable as a subprocess for H5J HEVC decoding and encoding.

This notice does not describe FFmpeg/libav libraries linked into OpenCV's wheel.
Those are bundled software with their own configuration and source/replacement
requirements; see `OpenCV-NOTICE.md`.
