# -*- coding: utf-8 -*-
"""Lazy managed FFmpeg resolution, validation, and installation for MADI3D.

The module is deliberately independent from Qt and performs no probing,
downloading, or subprocess execution at import time. H5J callers can use it on
demand to locate MADI3D's verified managed FFmpeg or to install its pinned,
verified managed binary after the UI has obtained user consent.
"""
from __future__ import annotations

import hashlib
import os
import platform
import ssl
import stat
import random
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib import error, request


FFMPEG_RELEASE = "n8.1.2-1"
FFMPEG_VERSION = "n8.1.2"
FFMPEG_RELEASE_BASE_URL = (
    "https://github.com/shaka-project/static-ffmpeg-binaries/releases/download/"
    f"{FFMPEG_RELEASE}"
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_FFMPEG_DOWNLOAD_BYTES = 128 * 1024 * 1024
H5J_FFMPEG_PROCESS_TIMEOUT = 30.0 * 60.0
FFMPEG_PROCESS_POLL_SECONDS = 0.10
# Inter-frame prediction in the pinned encoder can change grayscale voxels even
# with lossless=1. Independent frames pass the multi-frame byte-exact probe.
H5J_ENCODER_PARAMETERS = "lossless=1:log-level=error:keyint=1"


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

    @property
    def url(self) -> str:
        return f"{FFMPEG_RELEASE_BASE_URL}/{self.filename}"


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
        "ffmpeg-win-x64.exe",
        "4044b3924c977ad31229d504c5d5b8685f9553124fbaff6e9c99048b42830341",
    ),
    ("linux", "x86_64"): FFmpegAsset(
        "linux",
        "x86_64",
        "ffmpeg-linux-x64",
        "9eac5b2b5076db5ff853a6fa0dcd6b8de7d0cac8481eadda6c47cd935825f1ee",
    ),
    ("macos", "x86_64"): FFmpegAsset(
        "macos",
        "x86_64",
        "ffmpeg-osx-x64",
        "62c87854d851f202fc4a29bdda0fe7b6ebcddd37b863482ce1bdc81151b03fe4",
    ),
    ("macos", "arm64"): FFmpegAsset(
        "macos",
        "arm64",
        "ffmpeg-osx-arm64",
        "e7b9fcd97f95f333512d6e8b8ac24d9dbc08f189f36047695499bd7b57214b22",
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
            pixels = random.Random(35).randbytes(5 * 64 * 64)
            raw = temp / "channel.raw"
            raw.write_bytes(pixels)
            encoded = temp / "channel.h265"
            encode = run([
                "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray",
                "-video_size", "64x64", "-framerate", "1", "-i", os.fspath(raw),
                "-an", "-c:v", "libx265",
                "-x265-params", H5J_ENCODER_PARAMETERS,
                "-pix_fmt", "gray", "-f", "hevc", os.fspath(encoded),
            ])
            if int(getattr(encode, "returncode", 1)) != 0 or not encoded.is_file() or encoded.stat().st_size <= 0:
                return _process_text(encode)[-2000:] or "FFmpeg could not encode the H5J HEVC probe."

            decoded_pattern = temp / "decoded_%06d.tif"
            decode = run([
                "-y", "-v", "error", "-i", os.fspath(encoded),
                "-compression_algo", "raw", os.fspath(decoded_pattern),
            ])
            decoded = temp / "decoded_000001.tif"
            if int(getattr(decode, "returncode", 1)) != 0 or not decoded.is_file() or decoded.stat().st_size <= 0:
                return _process_text(decode)[-2000:] or "FFmpeg could not decode the H5J HEVC probe to raw TIFF."
            decoded_raw = temp / "decoded.raw"
            exact = run(["-y", "-v", "error", "-i", os.fspath(encoded),
                         "-pix_fmt", "gray", "-f", "rawvideo", os.fspath(decoded_raw)])
            if int(getattr(exact, "returncode", 1)) != 0 or not decoded_raw.is_file() or decoded_raw.read_bytes() != pixels:
                return "FFmpeg did not preserve the multi-frame grayscale probe exactly."

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


def run_ffmpeg_process(
    executable: os.PathLike | str,
    args,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    timeout: Optional[float] = H5J_FFMPEG_PROCESS_TIMEOUT,
    poll_interval: float = FFMPEG_PROCESS_POLL_SECONDS,
    popen: Callable = subprocess.Popen,
):
    """Run one FFmpeg command with bounded wait and cooperative cancellation."""
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

    if _managed_path_ready(
        destination, selected, runner=runner, max_bytes=max_bytes
    ):
        return os.fspath(destination)

    http_request = request.Request(
        selected.url,
        headers={"User-Agent": "MADI3D-FFmpeg-setup"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    content_length = 0

    try:
        context = _https_context()
        with (
            urlopen(http_request, timeout=60, context=context) as response,
            temporary.open("wb") as output,
        ):
            raw_length = response.headers.get("Content-Length", "")
            try:
                content_length = int(raw_length)
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
    except (FFmpegDownloadCancelled, FFmpegDownloadError):
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
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

    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != selected.sha256.lower():
        temporary.unlink(missing_ok=True)
        raise FFmpegDownloadError(
            "The downloaded FFmpeg file failed SHA-256 verification and was discarded."
        )

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
