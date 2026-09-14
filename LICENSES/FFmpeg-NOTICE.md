# FFmpeg notice for MADI3D

MADI3D does not bundle an FFmpeg executable. H5J import/export requires an
external FFmpeg executable with HEVC decoding and libx265 encoding support.

MADI3D uses its verified managed binary for H5J and does not discover FFmpeg on
the system PATH. If the managed binary is unavailable, the user may explicitly
choose to download the pinned managed binary. MADI3D stores it in the current
user's application-data directory, verifies it against a pinned SHA-256 checksum
before use, and does not add it to the system PATH.

## Managed FFmpeg build

Provider/build project: Shaka Project `static-ffmpeg-binaries`

Pinned release: `n8.1.2-1`

FFmpeg version: `n8.1.2`

Build/release source:
https://github.com/shaka-project/static-ffmpeg-binaries/tree/n8.1.2-1

The tag resolved to build-source revision
`88caac417541f3bb678fa6670cb73f2d74c7aaf9` when audited on 2026-09-13.
`versions.txt` and `build-scripts/` at that revision identify the component
versions, patches and build flags. All four managed asset SHA-256 values in
MADI3D's backend matched the provider's published asset digests on that date.

Release assets:
https://github.com/shaka-project/static-ffmpeg-binaries/releases/tag/n8.1.2-1

The provider builds FFmpeg and its dependencies from source. The pinned build
includes x265 and configures FFmpeg with `--enable-gpl --enable-version3`. The
provider states that the resulting FFmpeg binaries are published under the GPL.

FFmpeg project and source:
https://ffmpeg.org/
https://ffmpeg.org/download.html

x265 source used by the provider build:
https://bitbucket.org/multicoreware/x265_git/src/4.2/

The verified executable and source identities are recorded in `EXTERNAL-TOOLS.json`.

The GNU General Public License version 3 text distributed with MADI3D is in:
`LICENSES/ffmpeg-gplv3.txt`.

FFmpeg/x265 remain separate external software. MADI3D invokes the selected
FFmpeg executable as a subprocess for H5J HEVC decoding and encoding.

This notice does not describe FFmpeg/libav libraries linked into OpenCV's wheel.
Those are bundled software with their own configuration and source/replacement
requirements; see `OpenCV-NOTICE.md`.
