# FFmpeg notice for MADI3D

MADI3D does not bundle an FFmpeg executable. H5J import/export requires an
external FFmpeg executable with HEVC decoding and libx265 encoding support.

MADI3D first reuses a previously verified managed binary when one is present,
otherwise it uses a compatible FFmpeg already installed on the user's system.
If neither is available, the user may explicitly choose to download a pinned
managed binary. MADI3D stores that binary in the current user's application-data
directory, verifies it against a pinned SHA-256 checksum before use, and does not
add it to the system PATH.

## Managed FFmpeg build

Provider/build project: Shaka Project `static-ffmpeg-binaries`

Pinned release: `n8.1.2-1`

FFmpeg version: `n8.1.2`

Build/release source:
https://github.com/shaka-project/static-ffmpeg-binaries/tree/n8.1.2-1

Release assets:
https://github.com/shaka-project/static-ffmpeg-binaries/releases/tag/n8.1.2-1

The provider builds FFmpeg and its dependencies from source. The pinned build
includes x265 and configures FFmpeg with `--enable-gpl --enable-version3`. The
provider states that the resulting FFmpeg binaries are published under the GPL.

FFmpeg project and source:
https://ffmpeg.org/
https://ffmpeg.org/download.html

x265 source used by the provider build:
https://bitbucket.org/multicoreware/x265_git.git

The GNU General Public License version 3 text distributed with MADI3D is in:
`LICENSES/ffmpeg-gplv3.txt`.

FFmpeg/x265 remain separate external software. MADI3D invokes the selected
FFmpeg executable as a subprocess for H5J HEVC decoding and encoding.
