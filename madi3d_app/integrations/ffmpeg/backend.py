# -*- coding: utf-8 -*-
"""Lazy managed FFmpeg resolution, validation, and installation for MADI3D.

The module is deliberately independent from Qt and performs no probing,
downloading, or subprocess execution at import time. H5J callers can use it on
demand to locate MADI3D's verified managed FFmpeg or to install its pinned,
verified managed binary after the UI has obtained user consent.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import lzma
import os
import platform
import ssl
import stat
import random
import subprocess
import tempfile
import tarfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib import error, request

from madi3d_app.diagnostics import report_diagnostic


FFMPEG_RELEASE = "9.0.2-h5j12-2"
FFMPEG_VERSION = "9.0.2"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_FFMPEG_DOWNLOAD_BYTES = 256 * 1024 * 1024
H5J_FFMPEG_PROCESS_TIMEOUT = 30.0 * 60.0
FFMPEG_PROCESS_POLL_SECONDS = 0.10
# Inter-frame prediction in the pinned encoder can change grayscale voxels even
# with lossless=1. Independent frames pass the multi-frame byte-exact probe.
H5J_ENCODER_PARAMETERS = "lossless=1:log-level=error:keyint=1"
H5J_ENCODER_FILTER = "scale=in_range=full:out_range=full,format=yuv444p"
H5J_DECODER_FILTER = "format=pix_fmts=gray|gray9le|gray10le|gray12le|gray14le|gray16le"


class FFmpegError(RuntimeError):
    pass


class FFmpegUnavailableError(FFmpegError):
    pass


class FFmpegValidationError(FFmpegError):
    pass


class FFmpegDownloadError(FFmpegError):
    pass


class FFmpegDownloadCancelled(FFmpegDownloadError):
    pass


class FFmpegProcessCancelled(FFmpegError):
    pass


class FFmpegProcessTimeout(FFmpegError):
    pass


@dataclass(frozen=True)
class FFmpegAsset:
    platform: str
    architecture: str
    filename: str
    sha256: str
    url: str
    archive_sha256: str
    archive_format: str
    archive_member: str


@dataclass(frozen=True)
class FFmpegValidation:
    executable: str
    ready: bool
    version: str = ""
    has_hevc_decoder: bool = False
    has_libx265_encoder: bool = False
    h5j_pipeline_ready: bool = False
    error: str = ""


FFMPEG_ASSETS = {
    ("windows", "x86_64"): FFmpegAsset(
        "windows",
        "x86_64",
        "ffmpeg-windows-x86_64.zip",
        "513368c81583572ce1d09d5c7ff827feed1a63db352412a5001f4da3a1583d98",
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-win64-gpl-9.0.zip",
        "a0e45723c72141975f51d8666302e614711745f3102b704ca3f82c897a58d278",
        "zip",
        "ffmpeg-n9.0.2-17-g2a571b6068-win64-gpl-9.0/bin/ffmpeg.exe",
    ),
    ("linux", "x86_64"): FFmpegAsset(
        "linux",
        "x86_64",
        "ffmpeg-linux-x86_64.tar.xz",
        "0d709a0bed1a43f24cd2c420878c63afea4580671b6949ad19ed124572f3bc3d",
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0.tar.xz",
        "68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269",
        "tar.xz",
        "ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0/bin/ffmpeg",
    ),
    ("macos", "x86_64"): FFmpegAsset(
        "macos",
        "x86_64",
        "ffmpeg-macos-x86_64.zip",
        "b25689cf2211c0582d317e849912769a1f1c93904fe102c485b5364a425633d1",
        "https://ffmpeg.martin-riedl.de/download/macos/amd64/1789931006_9.0.2/ffmpeg.zip",
        "7c6b4125b191cbf773832dc51f424cf2b6bb7da43007d1e066f95909e47cacd4",
        "zip",
        "ffmpeg",
    ),
    ("macos", "arm64"): FFmpegAsset(
        "macos",
        "arm64",
        "ffmpeg-macos-arm64.zip",
        "2e11c6f90993cdb79fff84d3f90044d28316b310e75b3e030cfc9a54f2c9d384",
        "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffmpeg.zip",
        "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924",
        "zip",
        "ffmpeg",
    ),
}


def platform_name(value: Optional[str] = None) -> str:
    raw = str(value if value is not None else platform.system()).strip().lower()
    if raw in {"win32", "windows", "win"}:
        return "windows"
    if raw in {"darwin", "mac", "macos", "osx"}:
        return "macos"
    if raw.startswith("linux"):
        return "linux"
    return raw


def architecture_name(value: Optional[str] = None) -> str:
    raw = str(value if value is not None else platform.machine()).strip().lower()
    if raw in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if raw in {"arm64", "aarch64"}:
        return "arm64"
    return raw


def current_asset(
    *, platform_value: Optional[str] = None, architecture_value: Optional[str] = None
) -> FFmpegAsset:
    key = (platform_name(platform_value), architecture_name(architecture_value))
    asset = FFMPEG_ASSETS.get(key)
    if asset is None:
        raise FFmpegUnavailableError(
            "Automatic FFmpeg setup is not available for "
            f"{key[0] or 'unknown platform'} / {key[1] or 'unknown architecture'}."
        )
    return asset


def managed_storage_root(
    *,
    platform_value: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[os.PathLike | str] = None,
) -> Path:
    system = platform_name(platform_value)
    env = os.environ if environ is None else environ
    user_home = Path.home() if home is None else Path(home)
    if system == "windows":
        base = Path(env.get("LOCALAPPDATA") or user_home / "AppData" / "Local")
        return base / "MADI3D" / "FFmpeg"
    if system == "macos":
        return user_home / "Library" / "Application Support" / "MADI3D" / "FFmpeg"
    if system == "linux":
        base = Path(env.get("XDG_DATA_HOME") or user_home / ".local" / "share")
        return base / "MADI3D" / "FFmpeg"
    raise FFmpegUnavailableError(
        f"Automatic FFmpeg setup does not support platform '{system}'."
    )


def managed_executable_path(
    *,
    asset: Optional[FFmpegAsset] = None,
    root: Optional[os.PathLike | str] = None,
) -> Path:
    selected = asset or current_asset()
    storage = Path(root) if root is not None else managed_storage_root(
        platform_value=selected.platform
    )
    executable_name = "ffmpeg.exe" if selected.platform == "windows" else "ffmpeg"
    return storage / FFMPEG_RELEASE / selected.platform / selected.architecture / executable_name


def _sha256_path(path: os.PathLike | str, *, max_bytes: int = MAX_FFMPEG_DOWNLOAD_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > int(max_bytes):
                raise FFmpegValidationError(
                    f"FFmpeg file exceeds the {int(max_bytes)} byte safety limit."
                )
            digest.update(chunk)
    return digest.hexdigest()


def _process_text(proc) -> str:
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return "\n".join(part for part in (str(stdout).strip(), str(stderr).strip()) if part)


def _codec_list_contains(text: str, codec: str) -> bool:
    wanted = str(codec).strip().lower()
    for raw_line in str(text or "").splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 2 and parts[1].lower() == wanted:
            return True
    return False


def read_grayscale_y4m_header(stream):
    """Read FFmpeg's self-describing grayscale frame layout without rescaling."""
    header = stream.readline(1024)
    if not header.startswith(b"YUV4MPEG2 ") or not header.endswith(b"\n"):
        raise ValueError("Invalid decoded H5J frame header.")
    fields = {value[:1]: value[1:] for value in header.split()[1:]}
    pixel_format = fields.get(b"C")
    depths = {b"mono": 8, b"mono9": 9, b"mono10": 10, b"mono12": 12,
              b"mono14": 14, b"mono16": 16}
    if pixel_format not in depths:
        raise ValueError("H5J decoding did not produce supported grayscale frames.")
    width, height = int(fields[b"W"]), int(fields[b"H"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid decoded H5J frame dimensions.")
    return width, height, depths[pixel_format]


def probe_h5j_pipeline(
    executable: os.PathLike | str,
    *,
    runner: Callable = subprocess.run,
    timeout: float = 60.0,
) -> str:
    """Return an error string unless FFmpeg can execute MADI3D's H5J codec path."""
    path = os.fspath(executable)

    def run(args):
        return runner(
            [path, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="madi3d_ffmpeg_probe_") as temp_dir:
            temp = Path(temp_dir)
            probes = (
                (8, "gray", random.Random(35).randbytes(5 * 64 * 64)),
                (12, "gray12le", b"".join(
                    value.to_bytes(2, "little")
                    for value in (index % 4096 for index in range(5 * 64 * 64))
                )),
            )
            for depth, pixel_format, pixels in probes:
                raw = temp / f"channel-{depth}.raw"
                raw.write_bytes(pixels)
                encoded = temp / f"channel-{depth}.h265"
                encoder_filter = (
                    H5J_ENCODER_FILTER if depth == 8
                    else "format=gray12le"
                )
                encode = run([
                    "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", pixel_format,
                    "-video_size", "64x64", "-framerate", "1", "-i", os.fspath(raw),
                    "-an", "-vf", encoder_filter,
                    "-color_range", "pc", "-c:v", "libx265",
                    "-x265-params", H5J_ENCODER_PARAMETERS,
                    "-pix_fmt", "+yuv444p" if depth == 8 else "+gray12le",
                    "-f", "hevc", os.fspath(encoded),
                ])
                if (
                    int(getattr(encode, "returncode", 1)) != 0
                    or not encoded.is_file()
                    or encoded.stat().st_size <= 0
                ):
                    return _process_text(encode)[-2000:] or (
                        f"FFmpeg could not encode the H5J {depth}-bit HEVC probe."
                    )

                decoded = temp / f"decoded-{depth}.y4m"
                decode = run([
                    "-y", "-v", "error", "-i", os.fspath(encoded),
                    "-vf", H5J_DECODER_FILTER, "-strict", "-1",
                    "-fps_mode", "passthrough", "-f", "yuv4mpegpipe", os.fspath(decoded),
                ])
                if (
                    int(getattr(decode, "returncode", 1)) != 0
                    or not decoded.is_file()
                    or decoded.stat().st_size <= 0
                ):
                    return _process_text(decode)[-2000:] or (
                        f"FFmpeg could not decode the H5J {depth}-bit HEVC probe "
                        "to grayscale frames."
                    )
                with decoded.open("rb") as stream:
                    if read_grayscale_y4m_header(stream) != (64, 64, depth):
                        return f"FFmpeg changed the {depth}-bit grayscale probe dimensions or sample depth."
                    frame_bytes = 64 * 64 * (1 if depth == 8 else 2)
                    for offset in range(0, len(pixels), frame_bytes):
                        if (
                            stream.read(6) != b"FRAME\n"
                            or stream.read(frame_bytes) != pixels[offset:offset + frame_bytes]
                        ):
                            return (
                                f"FFmpeg did not preserve the multi-frame {depth}-bit "
                                "grayscale probe exactly."
                            )
                    if stream.read(1):
                        return f"FFmpeg produced unexpected extra {depth}-bit grayscale probe frames."

    except Exception as exc:
        return str(exc)
    return ""


def validate_ffmpeg(
    executable: os.PathLike | str,
    *,
    runner: Callable = subprocess.run,
    timeout: float = 20.0,
    functional_probe: Optional[Callable] = None,
) -> FFmpegValidation:
    path = os.fspath(executable)

    def run(args):
        return runner(
            [path, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    try:
        version_proc = run(["-version"])
        version_text = _process_text(version_proc)
        if int(getattr(version_proc, "returncode", 1)) != 0:
            return FFmpegValidation(
                executable=path,
                ready=False,
                error=version_text or "FFmpeg version check failed.",
            )
        version = next((line.strip() for line in version_text.splitlines() if line.strip()), "")

        decoder_proc = run(["-hide_banner", "-decoders"])
        decoder_text = _process_text(decoder_proc)
        has_hevc = (
            int(getattr(decoder_proc, "returncode", 1)) == 0
            and _codec_list_contains(decoder_text, "hevc")
        )

        encoder_proc = run(["-hide_banner", "-encoders"])
        encoder_text = _process_text(encoder_proc)
        has_x265 = (
            int(getattr(encoder_proc, "returncode", 1)) == 0
            and _codec_list_contains(encoder_text, "libx265")
        )
    except Exception as exc:
        return FFmpegValidation(executable=path, ready=False, error=str(exc))

    missing = []
    if not has_hevc:
        missing.append("HEVC decoder")
    if not has_x265:
        missing.append("libx265 encoder")

    pipeline_error = ""
    if not missing:
        probe = functional_probe
        if probe is None and runner is subprocess.run:
            probe = probe_h5j_pipeline
        if probe is not None:
            try:
                pipeline_error = str(probe(path, runner=runner, timeout=max(60.0, float(timeout))) or "")
            except TypeError:
                pipeline_error = str(probe(path) or "")
            except Exception as exc:
                pipeline_error = str(exc)

    ready = not missing and not pipeline_error
    if missing:
        error = "Missing required H5J capability: " + ", ".join(missing)
    elif pipeline_error:
        error = "FFmpeg failed the H5J TIFF/HEVC functional probe: " + pipeline_error
    else:
        error = ""
    return FFmpegValidation(
        executable=path,
        ready=ready,
        version=version,
        has_hevc_decoder=has_hevc,
        has_libx265_encoder=has_x265,
        h5j_pipeline_ready=not pipeline_error and not missing,
        error=error,
    )


def _stop_ffmpeg_process(process) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.communicate(timeout=2.0)
        return
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.communicate(timeout=2.0)
    except Exception:
        pass


#: Maximum characters of successful-run stderr surfaced to the Activity Log.
FFMPEG_DIAGNOSTIC_TAIL_CHARS = 16 * 1024


def _report_success_diagnostics(stderr, operation: str) -> None:
    """Surface a successful FFmpeg run's stderr through the diagnostic bridge.

    A failed run's output belongs to the caller's error path (the workflow
    logs the raised error with its own attribution), and FFmpeg stdout is
    machine data (raw frames, ``-progress`` protocol) that must never be
    logged.  Only the successful-run stderr would otherwise vanish silently.
    """
    if not stderr:
        return
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.strip()
    if not text:
        return
    if len(text) > FFMPEG_DIAGNOSTIC_TAIL_CHARS:
        text = "\u2026 [older output omitted]" + text[-FFMPEG_DIAGNOSTIC_TAIL_CHARS:]
    label = f"FFmpeg {operation}".strip()
    report_diagnostic(f"{label} reported:\n{text}", severity="Warning")


def run_ffmpeg_process(
    executable: os.PathLike | str,
    args,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    timeout: Optional[float] = H5J_FFMPEG_PROCESS_TIMEOUT,
    poll_interval: float = FFMPEG_PROCESS_POLL_SECONDS,
    popen: Callable = subprocess.Popen,
    operation: str = "",
):
    """Run one FFmpeg command with bounded wait and cooperative cancellation.

    ``operation`` labels the workflow step for log attribution. A
    successful run's stderr is surfaced as a bounded diagnostic; a failing
    run leaves its output to the caller's exception path so nothing is
    logged twice. stdout is never logged.
    """
    command = [os.fspath(executable), *[os.fspath(arg) for arg in args]]
    popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    if os.name == "nt" and popen is subprocess.Popen:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = popen(command, **popen_kwargs)
    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    poll_interval = max(0.02, float(poll_interval))
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=poll_interval)
                if int(process.returncode) == 0:
                    _report_success_diagnostics(stderr, operation)
                return subprocess.CompletedProcess(command, int(process.returncode), stdout, stderr)
            except subprocess.TimeoutExpired:
                if cancel_check is not None and cancel_check():
                    _stop_ffmpeg_process(process)
                    raise FFmpegProcessCancelled("FFmpeg operation was cancelled.")
                if deadline is not None and time.monotonic() >= deadline:
                    _stop_ffmpeg_process(process)
                    raise FFmpegProcessTimeout(
                        f"FFmpeg did not finish within {float(timeout):.0f} seconds."
                    )
    except Exception:
        if process.poll() is None:
            _stop_ffmpeg_process(process)
        raise


def _make_executable(path: Path) -> None:
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _managed_path_ready(
    path: Path,
    asset: FFmpegAsset,
    *,
    runner: Callable = subprocess.run,
    max_bytes: int = MAX_FFMPEG_DOWNLOAD_BYTES,
) -> bool:
    """Return True only for the exact pinned managed binary with H5J support."""
    if not path.is_file():
        return False
    try:
        if _sha256_path(path, max_bytes=max_bytes).lower() != asset.sha256.lower():
            return False
        _make_executable(path)
    except Exception:
        return False
    return bool(validate_ffmpeg(path, runner=runner).ready)


def _wait_for_managed_destination(
    path: Path,
    asset: FFmpegAsset,
    *,
    runner: Callable = subprocess.run,
    max_bytes: int = MAX_FFMPEG_DOWNLOAD_BYTES,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Let a concurrent installer finish promoting the same pinned binary."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        if _managed_path_ready(path, asset, runner=runner, max_bytes=max_bytes):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.01, float(poll_interval)))


def find_managed_ffmpeg(
    *,
    asset: Optional[FFmpegAsset] = None,
    root: Optional[os.PathLike | str] = None,
    runner: Callable = subprocess.run,
) -> Optional[str]:
    """Return verified managed FFmpeg, without PATH discovery, UI, or download."""
    selected = asset or current_asset()
    path = managed_executable_path(asset=selected, root=root)
    return os.fspath(path) if _managed_path_ready(path, selected, runner=runner) else None


def _https_context() -> ssl.SSLContext:
    """Use native CA trust without changing SSL behavior outside this download.

    In particular, macOS Keychain trust must not depend on the build Python's
    OpenSSL CA paths remaining available after freezing or on the user's Mac.
    Import lazily so offline discovery/reuse does not require TLS setup.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except (ImportError, OSError) as exc:
        raise FFmpegDownloadError(
            "MADI3D could not access this computer's trusted certificate authorities. "
            "Reinstall or update MADI3D, then try FFmpeg setup again. "
            "No FFmpeg file was installed."
        ) from exc


def _extract_pinned_archive_member(
    archive_path: Path,
    destination_path: Path,
    asset: FFmpegAsset,
    *,
    max_bytes: int,
    cancel_check: Optional[Callable[[], bool]],
) -> None:
    """Stream only the pinned regular-file member and verify its executable hash."""
    try:
        with ExitStack() as stack:
            if asset.archive_format == "zip":
                archive = stack.enter_context(zipfile.ZipFile(archive_path))
                matches = [
                    info for info in archive.infolist()
                    if info.filename == asset.archive_member
                ]
                if len(matches) != 1:
                    raise FFmpegDownloadError(
                        "The FFmpeg archive does not contain one exact pinned executable member."
                    )
                info = matches[0]
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if info.is_dir() or stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG):
                    raise FFmpegDownloadError(
                        "The pinned FFmpeg archive member is not a regular executable file."
                    )
                source = stack.enter_context(archive.open(info, "r"))
                expected_size = info.file_size
            elif asset.archive_format == "tar.xz":
                archive = stack.enter_context(tarfile.open(archive_path, mode="r:xz"))
                matches = [
                    info for info in archive.getmembers()
                    if info.name == asset.archive_member
                ]
                if len(matches) != 1:
                    raise FFmpegDownloadError(
                        "The FFmpeg archive does not contain one exact pinned executable member."
                    )
                info = matches[0]
                if not info.isfile():
                    raise FFmpegDownloadError(
                        "The pinned FFmpeg archive member is not a regular executable file."
                    )
                source = archive.extractfile(info)
                if source is None:
                    raise FFmpegDownloadError("The pinned FFmpeg archive member could not be read.")
                source = stack.enter_context(source)
                expected_size = info.size
            else:
                raise FFmpegDownloadError(
                    f"Unsupported pinned FFmpeg archive format: {asset.archive_format}"
                )

            if expected_size <= 0 or expected_size > int(max_bytes):
                raise FFmpegDownloadError("The FFmpeg executable exceeds the safety limit.")
            digest = hashlib.sha256()
            extracted = 0
            with destination_path.open("wb") as output:
                while True:
                    if cancel_check is not None and cancel_check():
                        raise FFmpegDownloadCancelled("FFmpeg download was cancelled.")
                    chunk = source.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    extracted += len(chunk)
                    if extracted > int(max_bytes) or extracted > expected_size:
                        raise FFmpegDownloadError(
                            "The FFmpeg executable exceeded its declared safety limit."
                        )
                    output.write(chunk)
                    digest.update(chunk)
            if extracted != expected_size:
                raise FFmpegDownloadError("The FFmpeg archive member was incomplete.")
            if digest.hexdigest().lower() != asset.sha256.lower():
                raise FFmpegDownloadError("The extracted FFmpeg executable failed SHA-256 verification.")
    except (FFmpegDownloadCancelled, FFmpegDownloadError):
        raise
    except (
        OSError, zipfile.BadZipFile, tarfile.TarError, lzma.LZMAError,
        RuntimeError, NotImplementedError,
    ) as exc:
        raise FFmpegDownloadError(f"The FFmpeg archive could not be read: {exc}") from exc


def install_managed_ffmpeg(
    *,
    asset: Optional[FFmpegAsset] = None,
    root: Optional[os.PathLike | str] = None,
    runner: Callable = subprocess.run,
    urlopen: Callable = request.urlopen,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    max_bytes: int = MAX_FFMPEG_DOWNLOAD_BYTES,
) -> str:
    selected = asset or current_asset()
    destination = managed_executable_path(asset=selected, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.part.{os.getpid()}.{uuid.uuid4().hex}"
    )
    archive_temporary = destination.with_name(
        f"{destination.name}.archive.part.{os.getpid()}.{uuid.uuid4().hex}"
    )

    if _managed_path_ready(
        destination, selected, runner=runner, max_bytes=max_bytes
    ):
        return os.fspath(destination)

    try:
        context = _https_context()
        for attempt in range(2):
            headers = {"User-Agent": "MADI3D-FFmpeg-setup"}
            if attempt:
                if cancel_check is not None and cancel_check():
                    raise FFmpegDownloadCancelled("FFmpeg download was cancelled.")
                headers["Cache-Control"] = "no-cache"
            http_request = request.Request(selected.url, headers=headers)
            digest = hashlib.sha256()
            downloaded = 0
            with (
                urlopen(http_request, timeout=60, context=context) as response,
                archive_temporary.open("wb") as output,
            ):
                try:
                    content_length = int(response.headers.get("Content-Length", ""))
                except (TypeError, ValueError):
                    content_length = 0
                if content_length > int(max_bytes):
                    raise FFmpegDownloadError(
                        "The FFmpeg download is unexpectedly larger than the safety limit."
                    )

                while True:
                    if cancel_check is not None and cancel_check():
                        raise FFmpegDownloadCancelled("FFmpeg download was cancelled.")
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > int(max_bytes):
                        raise FFmpegDownloadError(
                            "The FFmpeg download exceeded the safety limit."
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(downloaded, content_length)
            if digest.hexdigest() == selected.archive_sha256.lower():
                break
            archive_temporary.unlink()
        else:
            raise FFmpegDownloadError(
                "The downloaded FFmpeg archive failed SHA-256 verification and was discarded."
            )
    except (FFmpegDownloadCancelled, FFmpegDownloadError):
        temporary.unlink(missing_ok=True)
        archive_temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        archive_temporary.unlink(missing_ok=True)
        reason = exc.reason if isinstance(exc, error.URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise FFmpegDownloadError(
                "MADI3D could not establish a trusted HTTPS connection while downloading FFmpeg.\n\n"
                "The server certificate could not be verified using this computer's trusted "
                "certificate authorities. Check the network connection or institutional "
                "HTTPS/certificate configuration, then try again.\n\n"
                "No FFmpeg file was installed."
            ) from exc
        raise FFmpegDownloadError(f"FFmpeg could not be downloaded: {exc}") from exc

    try:
        _extract_pinned_archive_member(
            archive_temporary, temporary, selected, max_bytes=max_bytes,
            cancel_check=cancel_check,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        archive_temporary.unlink(missing_ok=True)
        raise
    archive_temporary.unlink(missing_ok=True)

    try:
        os.replace(temporary, destination)
        _make_executable(destination)
    except OSError as exc:
        # Windows will not replace an executable while another process is
        # validating/running it. If another MADI3D instance won the promotion
        # race, converge on that file only after the pinned hash and H5J
        # capability checks succeed. Never delete the winner.
        if _wait_for_managed_destination(
            destination, selected, runner=runner, max_bytes=max_bytes
        ):
            temporary.unlink(missing_ok=True)
            return os.fspath(destination)
        temporary.unlink(missing_ok=True)
        raise FFmpegDownloadError(f"FFmpeg could not be installed: {exc}") from exc
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise FFmpegDownloadError(f"FFmpeg could not be installed: {exc}") from exc

    validation = validate_ffmpeg(destination, runner=runner)
    if not validation.ready:
        destination.unlink(missing_ok=True)
        raise FFmpegValidationError(
            validation.error or "The downloaded FFmpeg binary failed validation."
        )
    return os.fspath(destination)


__all__ = [
    "DOWNLOAD_CHUNK_SIZE",
    "FFMPEG_ASSETS",
    "FFMPEG_RELEASE",
    "FFMPEG_VERSION",
    "H5J_FFMPEG_PROCESS_TIMEOUT",
    "MAX_FFMPEG_DOWNLOAD_BYTES",
    "FFmpegAsset",
    "FFmpegDownloadCancelled",
    "FFmpegDownloadError",
    "FFmpegProcessCancelled",
    "FFmpegProcessTimeout",
    "FFmpegError",
    "FFmpegUnavailableError",
    "FFmpegValidation",
    "FFmpegValidationError",
    "architecture_name",
    "current_asset",
    "find_managed_ffmpeg",
    "install_managed_ffmpeg",
    "managed_executable_path",
    "managed_storage_root",
    "platform_name",
    "probe_h5j_pipeline",
    "run_ffmpeg_process",
    "validate_ffmpeg",
]
