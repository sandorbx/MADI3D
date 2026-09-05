# -*- coding: utf-8 -*-
"""First-use CMTK setup UI and platform provisioning for MADI3D."""
from __future__ import annotations

import codecs
import hashlib
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib import request

from madi3d_app.integrations.cmtk.backend import (
    CMTK_CORE_TOOLS,
    CMTKSetupCancelled,
    CMTKSetupPending,
    CMTKUnavailableError,
    MANAGED_WSL_DISTRO,
    NativeCMTKBackend,
    WSLCMTKBackend,
    decode_console,
    get_cmtk_manager,
    process_error,
)


MANAGED_MARKER = "/etc/madi3d-cmtk-managed"
NATIVE_APT_GET = "/usr/bin/apt-get"
NATIVE_APT_CACHE = "/usr/bin/apt-cache"
NATIVE_ENV = "/usr/bin/env"
NATIVE_PKEXEC = "/usr/bin/pkexec"
UBUNTU_WSL_RELEASE = "Ubuntu 24.04.4 LTS"
UBUNTU_WSL_IMAGES = {
    "amd64": (
        "https://releases.ubuntu.com/24.04.4/ubuntu-24.04.4-wsl-amd64.wsl",
        "9b2f7730dc68227dd04a9f3e5eab86ad85caf556b8606ad94f1f29ff5c4fd3f5",
    ),
    "arm64": (
        "https://cdimages.ubuntu.com/releases/24.04.4/release/ubuntu-24.04.4-wsl-arm64.wsl",
        "6b244d89f412a68f51e58f396fab65bed3b5896a25c045a99bef9c78a07df507",
    ),
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_WSL_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


class _WSL2FeatureUnavailable(CMTKUnavailableError):
    """The WSL command exists, but its version-2 VM cannot be created."""


def _is_wsl2_feature_error(message):
    text = str(message or "").lower()
    return any(marker in text for marker in (
        "hcs_e_service_not_available",
        "hcs_e_hyperv_not_installed",
        "0x80370102",
        "required feature is not installed",
    ))


def _combined_output(proc):
    parts = [
        decode_console(getattr(proc, "stdout", b"")).strip(),
        decode_console(getattr(proc, "stderr", b"")).strip(),
    ]
    return "\n".join(p for p in parts if p).strip()


def _log_process_output(proc, log, max_lines=100):
    text = _combined_output(proc)
    if not text:
        return
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = ["..."] + lines[-max_lines:]
    for line in lines:
        log(line)


def _cancel_requested(cancel_check):
    return bool(cancel_check is not None and cancel_check())


def _stop_process(proc):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _console_stream_encoding(data):
    """Choose one decoder for a redirected Windows console byte stream."""
    data = bytes(data or b"")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if data.count(b"\x00") > max(2, len(data) // 8):
        even_nuls = data[0::2].count(b"\x00")
        odd_nuls = data[1::2].count(b"\x00")
        return "utf-16be" if even_nuls > odd_nuls else "utf-16le"
    return "utf-8"


def _decoded_stream_chunks(stream, chunk_size=4096):
    """Incrementally decode one console stream without splitting code units."""
    read = getattr(stream, "read1", None) or stream.read
    first = read(chunk_size)
    if not first:
        return
    decoder = codecs.getincrementaldecoder(
        _console_stream_encoding(first)
    )(errors="replace")
    text = decoder.decode(first)
    if text:
        yield text
    while True:
        chunk = read(chunk_size)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if text:
            yield text
    final = decoder.decode(b"", final=True)
    if final:
        yield final


def _stream_real_command(cmd, log, *, timeout, description, cancel_check, allow_terminate):
    """Run a real setup subprocess with live combined stdout/stderr."""
    try:
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except Exception as exc:
        raise CMTKUnavailableError(f"{description} could not be started: {exc}") from exc

    output_queue = queue.Queue()
    sentinel = object()

    def read_output():
        try:
            stream = proc.stdout
            if stream is not None:
                for text in _decoded_stream_chunks(stream):
                    output_queue.put(text)
        finally:
            output_queue.put(sentinel)

    threading.Thread(target=read_output, daemon=True).start()
    output = []
    reader_finished = False
    deadline = time.monotonic() + float(timeout)
    protected_timeout_reported = False

    while True:
        try:
            item = output_queue.get(timeout=0.1)
            if item is sentinel:
                reader_finished = True
            else:
                text = str(item).rstrip()
                if text:
                    output.append(text)
                    for line in text.splitlines():
                        log(line)
        except queue.Empty:
            pass

        if _cancel_requested(cancel_check) and allow_terminate:
            _stop_process(proc)
            raise CMTKSetupCancelled(f"{description} was cancelled.")

        if time.monotonic() >= deadline and proc.poll() is None:
            if allow_terminate:
                _stop_process(proc)
                raise CMTKUnavailableError(
                    f"{description} timed out after {int(timeout)} seconds."
                )
            if not protected_timeout_reported:
                log(
                    f"{description} exceeded {int(timeout)} seconds and is still running. "
                    "MADI3D will not interrupt this protected package-manager step."
                )
                protected_timeout_reported = True
            deadline = float("inf")

        if proc.poll() is not None and reader_finished and output_queue.empty():
            break

    if proc.stdout is not None:
        proc.stdout.close()
    completed = subprocess.CompletedProcess(
        list(cmd), proc.returncode, stdout="\n".join(output), stderr=""
    )
    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled(
            "Cancellation requested; the current protected setup step finished safely."
        )
    return completed


def _run_command(
    manager,
    cmd,
    log,
    *,
    timeout=1200,
    description="Command",
    cancel_check=None,
    allow_terminate=True,
):
    log("$ " + " ".join(str(v) for v in cmd))
    log(f"{description} is running...")
    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")

    try:
        if manager.runner is subprocess.run:
            proc = _stream_real_command(
                cmd,
                log,
                timeout=timeout,
                description=description,
                cancel_check=cancel_check,
                allow_terminate=allow_terminate,
            )
        else:
            proc = manager.runner(
                list(cmd), capture_output=True, check=False, timeout=timeout
            )
            _log_process_output(proc, log)
            if _cancel_requested(cancel_check):
                raise CMTKSetupCancelled(
                    "Cancellation requested; the current setup step finished safely."
                )
    except (CMTKSetupCancelled, CMTKUnavailableError):
        raise
    except Exception as exc:
        raise CMTKUnavailableError(f"{description} could not be started: {exc}") from exc

    if getattr(proc, "returncode", 1) != 0:
        raise CMTKUnavailableError(
            process_error(proc) or f"{description} failed with exit code {proc.returncode}."
        )
    return proc


def _run_elevated_wsl_install(
    manager, args, log, *, timeout=1200, cancel_check=None
):
    """Run one WSL install command elevated and wait for its process to finish."""
    powershell = manager.which("powershell.exe") or manager.which("powershell") or "powershell.exe"
    quoted = ", ".join("'" + str(v).replace("'", "''") + "'" for v in args)
    script = (
        "$p = Start-Process -FilePath 'wsl.exe' "
        f"-ArgumentList @({quoted}) -Verb RunAs -Wait -PassThru; "
        "exit $p.ExitCode"
    )
    log("Windows administrator approval is required for this setup step.")
    return _run_command(
        manager,
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        log,
        timeout=timeout,
        description="Elevated WSL installation",
        cancel_check=cancel_check,
        # Terminating the outer PowerShell process can leave the elevated child
        # running. Honour cancellation immediately after this protected step.
        allow_terminate=False,
    )


def _distro_is_installed(manager, name):
    target = str(name or "").lower()
    return any(str(d).lower() == target for d in manager.list_wsl_distros())


def _managed_marker_exists(manager, distro):
    proc = manager.capture(
        ["wsl.exe", "-d", distro, "-u", "root", "--exec", "test", "-f", MANAGED_MARKER],
        timeout=30,
    )
    return proc is not None and getattr(proc, "returncode", 1) == 0


def _windows_wsl_architecture():
    machine = str(
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or platform.machine()
    ).strip().lower()
    if machine in ("amd64", "x86_64"):
        return "amd64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    raise CMTKUnavailableError(
        f"Automatic CMTK setup does not have an Ubuntu WSL image for architecture '{machine}'."
    )


def _managed_storage_root(manager):
    local_app_data = str(os.environ.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        return Path(local_app_data) / "MADI3D" / "CMTK"
    try:
        return manager.config_path().parent / "CMTK"
    except Exception as exc:
        raise CMTKUnavailableError(
            "Could not determine a persistent local folder for the isolated CMTK environment."
        ) from exc


def _ubuntu_image_cache_path(manager):
    return (
        _managed_storage_root(manager)
        / "Downloads"
        / f"ubuntu-24.04.4-{_windows_wsl_architecture()}.wsl"
    )


def _remove_cached_ubuntu_image(manager, log):
    try:
        image_path = _ubuntu_image_cache_path(manager)
        if image_path.is_file():
            image_path.unlink()
            log("Removed the downloaded Ubuntu image after successful setup.")
    except Exception as exc:
        log(f"WARNING: Could not remove the cached Ubuntu image: {exc}")


def _sha256_path(path, *, cancel_check=None):
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as source:
        while True:
            if _cancel_requested(cancel_check):
                raise CMTKSetupCancelled("Ubuntu WSL image verification was cancelled.")
            chunk = source.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_WSL_IMAGE_BYTES:
                raise CMTKUnavailableError(
                    "The cached Ubuntu WSL image exceeded the 2 GiB safety limit."
                )
            digest.update(chunk)
    return digest.hexdigest()


def _download_official_ubuntu_image(
    destination, log, *, cancel_check=None, urlopen=request.urlopen
):
    architecture = _windows_wsl_architecture()
    url, expected_sha256 = UBUNTU_WSL_IMAGES[architecture]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if destination.exists() and not destination.is_file():
        raise CMTKUnavailableError(
            f"The Ubuntu image cache path is not a file: {destination}"
        )

    if destination.is_file():
        log(f"Checking cached {UBUNTU_WSL_RELEASE} WSL image.")
        try:
            cached_sha256 = _sha256_path(
                destination, cancel_check=cancel_check
            )
        except CMTKSetupCancelled:
            raise
        except Exception as exc:
            log(f"Cached Ubuntu image could not be verified and will be replaced: {exc}")
            destination.unlink(missing_ok=True)
        else:
            if cached_sha256.lower() == expected_sha256.lower():
                log(f"Using cached verified Ubuntu WSL image: {destination}")
                log(f"Verified Ubuntu WSL image SHA-256: {cached_sha256}")
                return destination
            log("Cached Ubuntu image failed SHA-256 verification and was discarded.")
            destination.unlink(missing_ok=True)

    temporary.unlink(missing_ok=True)
    log(f"Downloading official {UBUNTU_WSL_RELEASE} WSL image for {architecture}.")
    log(url)
    digest = hashlib.sha256()
    downloaded = 0
    next_report = 32 * 1024 * 1024
    http_request = request.Request(url, headers={"User-Agent": "MADI3D-CMTK-setup"})

    try:
        with urlopen(http_request, timeout=60) as response, temporary.open("wb") as output:
            raw_length = response.headers.get("Content-Length", "")
            try:
                content_length = int(raw_length)
            except (TypeError, ValueError):
                content_length = 0
            if content_length > MAX_WSL_IMAGE_BYTES:
                raise CMTKUnavailableError(
                    "The official Ubuntu WSL image is unexpectedly larger than 2 GiB."
                )

            while True:
                if _cancel_requested(cancel_check):
                    raise CMTKSetupCancelled("Ubuntu WSL image download was cancelled.")
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_WSL_IMAGE_BYTES:
                    raise CMTKUnavailableError(
                        "The official Ubuntu WSL image exceeded the 2 GiB safety limit."
                    )
                output.write(chunk)
                digest.update(chunk)
                if downloaded >= next_report:
                    if content_length:
                        percent = min(100, int(downloaded * 100 / content_length))
                        log(f"Downloaded {downloaded // (1024 * 1024)} MiB ({percent}%).")
                    else:
                        log(f"Downloaded {downloaded // (1024 * 1024)} MiB.")
                    next_report += 32 * 1024 * 1024
    except (CMTKSetupCancelled, CMTKUnavailableError):
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise CMTKUnavailableError(
            f"The official Ubuntu WSL image could not be downloaded: {exc}"
        ) from exc

    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        temporary.unlink(missing_ok=True)
        raise CMTKUnavailableError(
            "The downloaded Ubuntu WSL image failed SHA-256 verification. "
            "The file was discarded and was not imported."
        )
    os.replace(temporary, destination)
    log(f"Verified Ubuntu WSL image SHA-256: {actual_sha256}")
    return destination


def _perform_distro_import(
    manager, image_path, target, install_path, log, *, cancel_check=None
):
    """Register a verified root filesystem under MADI3D's reserved distro name."""
    install_path = Path(install_path)
    install_path.parent.mkdir(parents=True, exist_ok=True)
    if install_path.exists():
        if not install_path.is_dir() or any(install_path.iterdir()):
            raise CMTKUnavailableError(
                f"The managed WSL storage folder already contains data: {install_path}\n"
                "MADI3D will not overwrite it. Remove that leftover folder or choose an "
                "existing CMTK installation."
            )
    else:
        install_path.mkdir()

    try:
        _run_command(
            manager,
            [
                "wsl.exe", "--import", target, str(install_path),
                str(Path(image_path)), "--version", "2",
            ],
            log,
            timeout=1200,
            description=f"Isolated WSL distribution import ({target})",
            cancel_check=cancel_check,
            allow_terminate=True,
        )
    except CMTKUnavailableError as exc:
        if not _distro_is_installed(manager, target):
            shutil.rmtree(install_path, ignore_errors=True)
        if _is_wsl2_feature_error(exc):
            raise _WSL2FeatureUnavailable(str(exc)) from exc
        raise
    except Exception:
        if not _distro_is_installed(manager, target):
            shutil.rmtree(install_path, ignore_errors=True)
        raise

    if not _distro_is_installed(manager, target):
        shutil.rmtree(install_path, ignore_errors=True)
        raise CMTKUnavailableError(
            f"Windows returned from the WSL import command, but distribution '{target}' "
            "was not registered."
        )
    return target


def _repair_wsl2_features(
    manager, log, stage, original_error, *, cancel_check=None
):
    config = manager.load_config()
    if config.get("wsl2_feature_repair_attempted"):
        raise CMTKUnavailableError(
            "WSL 2 still cannot create its virtual machine after Windows feature repair. "
            "Enable hardware virtualization (Intel VT-x/AMD-V/SVM) in BIOS/UEFI, ensure "
            "Virtual Machine Platform is enabled in Windows Features, and ensure the Windows "
            "hypervisor is allowed to start. Then restart Windows and retry.\n\n"
            f"Original WSL error:\n{original_error}"
        )

    stage("Enabling required WSL 2 virtualization features")
    log(
        "WSL is installed, but Windows could not create a WSL 2 virtual machine. "
        "MADI3D will ask Windows to enable the required components without installing "
        "or modifying any Linux distribution."
    )
    manager.update_config(setup_mode="auto", setup_pending=True)
    _run_elevated_wsl_install(
        manager,
        ["--install", "--no-distribution"],
        log,
        timeout=1200,
        cancel_check=cancel_check,
    )
    manager.update_config(
        setup_mode="auto",
        setup_pending=True,
        wsl2_feature_repair_attempted=True,
    )
    raise CMTKSetupPending(
        "Windows enabled or repaired the WSL 2 virtualization components. Restart Windows, "
        "then retry the CMTK action; MADI3D will resume automatically and reuse the verified "
        "Ubuntu download. If WSL 2 still cannot start after the restart, enable hardware "
        "virtualization (Intel VT-x/AMD-V/SVM) in BIOS/UEFI."
    )


def _install_managed_distribution(manager, log, stage, cancel_check=None):
    """Create/use a MADI-owned WSL distro without changing a personal Ubuntu distro."""
    config = manager.load_config()

    configured = str(config.get("distro") or "")
    configured_managed = bool(config.get("managed_distro"))
    setup_pending = bool(config.get("setup_pending"))

    # Earlier draft builds could incorrectly record the generic "Ubuntu" distro
    # as managed. Never honour that unsafe state, even when setup_pending is set.
    if (
        configured_managed
        and configured
        and configured.lower() != MANAGED_WSL_DISTRO.lower()
    ):
        log(
            f"Ignoring unsafe legacy managed-distro state for '{configured}'. "
            f"Only '{MANAGED_WSL_DISTRO}' can be managed automatically."
        )
        manager.update_config(
            managed_distro=None,
            distro=None,
            user=None,
            managed_source=None,
            managed_install_path=None,
            selection_source=None,
            cmtk_package_version=None,
        )
        configured = ""
        configured_managed = False

    if configured_managed and configured and _distro_is_installed(manager, configured):
        if setup_pending or _managed_marker_exists(manager, configured):
            log(f"Resuming MADI3D-managed WSL distribution: {configured}")
            return configured
        raise CMTKUnavailableError(
            f"MADI3D has stale ownership metadata for WSL distribution '{configured}', but the "
            "managed marker is missing. Automatic setup will not modify that distribution. "
            "Choose an existing installation explicitly or remove the stale MADI3D CMTK settings."
        )

    if _distro_is_installed(manager, MANAGED_WSL_DISTRO):
        if _managed_marker_exists(manager, MANAGED_WSL_DISTRO):
            manager.update_config(
                managed_distro=True, distro=MANAGED_WSL_DISTRO, user="root"
            )
            log(f"Using existing MADI3D-managed WSL distribution: {MANAGED_WSL_DISTRO}")
            return MANAGED_WSL_DISTRO
        raise CMTKUnavailableError(
            f"A WSL distribution named '{MANAGED_WSL_DISTRO}' already exists, but MADI3D cannot "
            "verify that it owns that environment. Automatic setup will not modify it. Rename/remove "
            "that distribution or choose 'Use an existing CMTK installation'."
        )

    target = MANAGED_WSL_DISTRO
    source = UBUNTU_WSL_RELEASE
    storage_root = _managed_storage_root(manager)
    install_path = storage_root / "WSL" / target
    download_root = storage_root / "Downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    image_path = _ubuntu_image_cache_path(manager)

    manager.update_config(
        setup_mode="auto",
        setup_pending=True,
        managed_distro=True,
        distro=target,
        user="root",
        managed_source=source,
        managed_install_path=str(install_path),
    )
    try:
        stage(f"Downloading {source} for isolated CMTK setup")
        image_path = _download_official_ubuntu_image(
            image_path,
            log,
            cancel_check=cancel_check,
        )
        stage("Importing isolated Ubuntu environment for CMTK")
        log(
            f"Importing the verified image as WSL distribution '{target}'. "
            "Existing WSL distributions are not modified."
        )
        try:
            distro = _perform_distro_import(
                manager,
                image_path,
                target,
                install_path,
                log,
                cancel_check=cancel_check,
            )
        except _WSL2FeatureUnavailable as exc:
            _repair_wsl2_features(
                manager,
                log,
                stage,
                exc,
                cancel_check=cancel_check,
            )
        _remove_cached_ubuntu_image(manager, log)
        manager.update_config(wsl2_feature_repair_attempted=None)
        return distro
    except Exception:
        if not _distro_is_installed(manager, target):
            manager.update_config(
                managed_distro=None,
                distro=None,
                user=None,
                managed_source=None,
                managed_install_path=None,
            )
        raise


def _install_wsl_if_missing(manager, log, stage, cancel_check=None):
    """Install WSL components without installing or claiming a default distribution."""
    stage("Installing Windows Subsystem for Linux")
    log(
        "WSL is not currently ready. MADI3D will enable/install WSL without "
        "installing a default Linux distribution."
    )
    # Persist resumable state before making any system change. If this cannot be
    # saved, stop rather than creating an environment MADI3D cannot later identify.
    manager.update_config(setup_mode="auto", setup_pending=True)
    _run_elevated_wsl_install(
        manager,
        ["--install", "--no-distribution"],
        log,
        timeout=1200,
        cancel_check=cancel_check,
    )

    if not manager.wsl_available():
        raise CMTKSetupPending(
            "Windows installed or enabled WSL components, but WSL is not ready yet. Restart Windows. "
            "On the next CMTK action, MADI3D will resume automatic setup without asking you to choose "
            "the setup mode again."
        )


def _automatic_windows_setup(
    manager, tools, log, stage, cancel_check=None, *, reinstall=False
):
    if not manager.wsl_available():
        _install_wsl_if_missing(
            manager, log, stage, cancel_check=cancel_check
        )

    distro = _install_managed_distribution(
        manager, log, stage, cancel_check=cancel_check
    )

    stage(f"Preparing {distro}")
    log("Initializing the isolated WSL distribution non-interactively as root.")
    _run_command(
        manager,
        ["wsl.exe", "-d", distro, "-u", "root", "--exec", "sh", "-lc", "printf MADI3D_WSL_READY"],
        log,
        timeout=300,
        description="WSL initialization",
        cancel_check=cancel_check,
        allow_terminate=True,
    )
    _run_command(
        manager,
        [
            "wsl.exe", "-d", distro, "-u", "root", "--exec", "sh", "-lc",
            f"printf '%s\\n' 'MADI3D managed CMTK backend' > {MANAGED_MARKER}",
        ],
        log,
        timeout=60,
        description="Marking managed WSL environment",
        cancel_check=cancel_check,
        allow_terminate=True,
    )

    stage("Waiting for Ubuntu initialization")
    _run_command(
        manager,
        [
            "wsl.exe", "-d", distro, "-u", "root", "--exec", "sh", "-lc",
            "if command -v cloud-init >/dev/null 2>&1; then "
            "cloud-init status --wait || echo 'cloud-init finished with a non-zero status'; fi",
        ],
        log,
        timeout=900,
        description="Ubuntu first-boot initialization",
        cancel_check=cancel_check,
        allow_terminate=True,
    )

    stage("Checking Ubuntu package state")
    _run_command(
        manager,
        [
            "wsl.exe", "-d", distro, "-u", "root", "--exec",
            "env", "DEBIAN_FRONTEND=noninteractive", "dpkg", "--configure", "-a",
        ],
        log,
        timeout=1200,
        description="dpkg recovery/configuration",
        cancel_check=cancel_check,
        allow_terminate=False,
    )

    stage("Updating Ubuntu package information")
    _run_command(
        manager,
        ["wsl.exe", "-d", distro, "-u", "root", "--exec", "apt-get", "-o", "DPkg::Lock::Timeout=300", "update"],
        log,
        timeout=1200,
        description="apt-get update",
        cancel_check=cancel_check,
        allow_terminate=True,
    )

    stage("Reinstalling CMTK" if reinstall else "Installing CMTK")
    log(
        ("Reinstalling" if reinstall else "Installing")
        + " the Ubuntu CMTK package in the isolated MADI3D environment."
    )
    install_args = [
        "apt-get", "-o", "DPkg::Lock::Timeout=300", "install", "-y",
    ]
    if reinstall:
        install_args.append("--reinstall")
    install_args.append("cmtk")
    _run_command(
        manager,
        [
            "wsl.exe", "-d", distro, "-u", "root", "--exec",
            "env", "DEBIAN_FRONTEND=noninteractive",
            *install_args,
        ],
        log,
        timeout=1800,
        description="CMTK package installation",
        cancel_check=cancel_check,
        allow_terminate=False,
    )

    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")
    version_proc = manager.capture(
        [
            "wsl.exe", "-d", distro, "-u", "root", "--exec",
            "dpkg-query", "-W", "-f=${Version}", "cmtk",
        ],
        timeout=30,
    )
    if version_proc is not None and getattr(version_proc, "returncode", 1) == 0:
        version = decode_console(getattr(version_proc, "stdout", b"")).strip()
        if version:
            log(f"Installed Ubuntu CMTK package version: {version}")
            manager.update_config(cmtk_package_version=version)

    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")
    stage("Validating CMTK tools")
    backend = WSLCMTKBackend(distro, runner=manager.runner, user="root")
    manager.update_config(managed_distro=True, distro=distro, user="root")
    if manager.validate_backend(backend, tools, managed=True) is None:
        raise CMTKUnavailableError(
            "CMTK installation completed, but one or more required CMTK tools failed validation.\n\n"
            + manager.last_status.diagnostic_text()
        )
    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")
    log("CMTK validation passed: " + ", ".join(tools))
    _remove_cached_ubuntu_image(manager, log)
    return backend


def _linux_distribution_id():
    try:
        for line in Path("/etc/os-release").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower()
    except Exception:
        pass
    return ""


def _system_executable_available(path):
    return Path(path).is_file() and os.access(path, os.X_OK)


def _native_setup_unavailable_reason(manager):
    if manager.platform_name() != "linux":
        return "Automatic native package installation is available only on Linux."

    distro = _linux_distribution_id()
    if distro not in ("ubuntu", "debian"):
        label = distro or "unknown Linux distribution"
        return (
            "Automatic CMTK installation is supported only on native Ubuntu/Debian; "
            f"this system reports {label}."
        )

    required = (
        (NATIVE_APT_GET, "apt-get"),
        (NATIVE_APT_CACHE, "apt-cache"),
        (NATIVE_ENV, "env"),
    )
    missing = [
        label
        for path, label in required
        if not _system_executable_available(path)
    ]
    if missing:
        return (
            "Required system package tools are unavailable: "
            + ", ".join(missing)
            + "."
        )

    running_as_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if not running_as_root and not _system_executable_available(NATIVE_PKEXEC):
        return (
            "Graphical administrator elevation (pkexec) is unavailable. "
            "Install the cmtk package manually."
        )
    return ""


def _apt_policy_has_candidate(text):
    for line in str(text or "").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "candidate":
            candidate = value.strip().lower()
            return candidate not in ("", "none", "(none)")
    return False


def _automatic_native_setup(
    manager, tools, log, stage, cancel_check=None, *, reinstall=False
):
    unavailable_reason = _native_setup_unavailable_reason(manager)
    if unavailable_reason:
        raise CMTKUnavailableError(
            unavailable_reason + " Use an existing CMTK installation instead."
        )

    log(
        "Native Linux automatic setup installs the distribution's system "
        "CMTK package."
    )
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        prefix = []
    else:
        prefix = [NATIVE_PKEXEC]

    stage("Updating package information")
    _run_command(
        manager,
        prefix + [
            NATIVE_ENV,
            "LC_ALL=C",
            NATIVE_APT_GET,
            "-o",
            "DPkg::Lock::Timeout=300",
            "update",
        ],
        log,
        timeout=1200,
        description="apt-get update",
        cancel_check=cancel_check,
        allow_terminate=False,
    )
    stage("Checking CMTK package availability")
    policy = _run_command(
        manager,
        [NATIVE_ENV, "LC_ALL=C", NATIVE_APT_CACHE, "policy", "cmtk"],
        log,
        timeout=60,
        description="CMTK package availability check",
        cancel_check=cancel_check,
        allow_terminate=True,
    )
    if not _apt_policy_has_candidate(_combined_output(policy)):
        raise CMTKUnavailableError(
            "The cmtk package has no installation candidate in the configured repositories. "
            "On Ubuntu, enable the official Universe repository and retry, or choose an "
            "existing CMTK installation. MADI3D did not change repository configuration."
        )
    stage("Reinstalling CMTK" if reinstall else "Installing CMTK")
    install_args = [
        NATIVE_ENV, "LC_ALL=C", "DEBIAN_FRONTEND=noninteractive",
        NATIVE_APT_GET, "-o", "DPkg::Lock::Timeout=300",
        "install", "-y",
    ]
    if reinstall:
        install_args.append("--reinstall")
    install_args.append("cmtk")
    _run_command(
        manager,
        prefix + install_args,
        log,
        timeout=1800,
        description="CMTK package installation",
        cancel_check=cancel_check,
        allow_terminate=False,
    )
    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")
    stage("Validating CMTK tools")
    manager.invalidate()
    backend = manager.detect(tools)
    if backend is None:
        raise CMTKUnavailableError(
            "CMTK installation completed but validation still failed.\n\n"
            + manager.last_status.diagnostic_text()
        )
    if _cancel_requested(cancel_check):
        raise CMTKSetupCancelled("CMTK setup was cancelled.")
    log("CMTK validation passed: " + ", ".join(tools))
    return backend


def _choose_existing_backend(manager, parent):
    from PySide6 import QtWidgets

    if manager.platform_name() == "windows":
        distros = manager.list_wsl_distros()
        if not distros:
            raise CMTKUnavailableError(
                "WSL is installed but no Linux distribution exists yet. "
                "Choose Automatic setup to install an isolated Ubuntu environment."
            )
        distro, ok = QtWidgets.QInputDialog.getItem(
            parent,
            "Use existing CMTK",
            "WSL distribution containing CMTK:",
            manager.rank_distros(distros),
            0,
            False,
        )
        if not ok or not distro:
            raise CMTKSetupCancelled()
        backend = WSLCMTKBackend(str(distro), runner=manager.runner)
    else:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Choose the CMTK launcher",
            "/usr/bin/cmtk" if manager.platform_name() == "linux" else "",
        )
        if not path:
            raise CMTKSetupCancelled()
        backend = NativeCMTKBackend(runner=manager.runner, cmtk_command=path)

    return backend


def _setup_dialog(manager, tools, parent=None, *, force_setup=False):
    """Detect/configure CMTK without blocking Qt; force_setup bypasses detection."""
    from PySide6 import QtCore, QtWidgets

    class Worker(QtCore.QObject):
        stage = QtCore.Signal(str)
        log = QtCore.Signal(str)
        completed = QtCore.Signal(object)
        not_found = QtCore.Signal()
        failed = QtCore.Signal(str)
        restart_required = QtCore.Signal(str)
        cancelled = QtCore.Signal(str)

        def __init__(self, mode, cancel_check, candidate=None, reinstall=False):
            super().__init__()
            self.mode = str(mode)
            self.cancel_check = cancel_check
            self.candidate = candidate
            self.reinstall = bool(reinstall)

        def run(self):
            try:
                if self.mode == "detect":
                    self.stage.emit("Checking for an existing CMTK installation")
                    backend = manager.detect(tools)
                    if _cancel_requested(self.cancel_check):
                        raise CMTKSetupCancelled("CMTK check was cancelled.")
                    if backend is None:
                        self.not_found.emit()
                        return
                elif self.mode == "existing":
                    self.stage.emit("Validating the selected CMTK installation")
                    backend = manager.validate_backend(
                        self.candidate, tools, managed=False
                    )
                    if _cancel_requested(self.cancel_check):
                        raise CMTKSetupCancelled("CMTK validation was cancelled.")
                    if backend is None:
                        raise CMTKUnavailableError(
                            "The selected installation could not run all required "
                            "CMTK tools.\n\n"
                            + manager.last_status.diagnostic_text()
                        )
                elif manager.platform_name() == "windows":
                    backend = _automatic_windows_setup(
                        manager,
                        tools,
                        self.log.emit,
                        self.stage.emit,
                        cancel_check=self.cancel_check,
                        reinstall=self.reinstall,
                    )
                else:
                    backend = _automatic_native_setup(
                        manager,
                        tools,
                        self.log.emit,
                        self.stage.emit,
                        cancel_check=self.cancel_check,
                        reinstall=self.reinstall,
                    )
            except CMTKSetupPending as exc:
                self.restart_required.emit(str(exc))
            except CMTKSetupCancelled as exc:
                self.cancelled.emit(str(exc) or "CMTK setup was cancelled.")
            except Exception as exc:
                self.failed.emit(str(exc))
            else:
                self.completed.emit(backend)

    class SetupDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.backend = None
            self.restart_message = ""
            self._running = False
            self._thread = None
            self._worker = None
            self._active_mode = ""
            self._terminal_kind = None
            self._terminal_payload = None
            self._cancel_requested_flag = False
            self._force_setup = bool(force_setup)
            config = manager.load_config()
            self._reinstall_requested = bool(
                self._force_setup
                and (config.get("last_successful_validation") or config.get("cmtk_package_version"))
            )
            self._resume_auto = bool(
                config.get("setup_mode") == "auto"
                and config.get("setup_pending")
            )

            self.setWindowTitle("CMTK setup")
            self.setModal(True)
            self.setSizeGripEnabled(True)
            self.resize(680, 500)
            if parent is not None:
                try:
                    self.setPalette(parent.palette())
                    self.setFont(parent.font())
                except Exception:
                    pass

            root = QtWidgets.QVBoxLayout(self)
            self.stack = QtWidgets.QStackedWidget(self)
            root.addWidget(self.stack, 1)

            select_page = QtWidgets.QWidget(self)
            select_layout = QtWidgets.QVBoxLayout(select_page)
            intro = QtWidgets.QLabel(
                "Install or repair the CMTK dependency used by MADI3D."
                if self._force_setup else
                "This feature needs CMTK. Detection and validation begin only "
                "after a CMTK action is requested."
            )
            intro.setWordWrap(True)
            select_layout.addWidget(intro)

            status_group = QtWidgets.QGroupBox("Detected status", select_page)
            status_layout = QtWidgets.QVBoxLayout(status_group)
            passive_status = manager.persisted_status() if self._force_setup else manager.last_status
            self.detected_status = QtWidgets.QLabel(
                passive_status.diagnostic_text()
                or "CMTK has not been checked yet."
            )
            self.detected_status.setWordWrap(True)
            self.detected_status.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            status_layout.addWidget(self.detected_status)
            select_layout.addWidget(status_group)

            mode_group = QtWidgets.QGroupBox("Setup method", select_page)
            mode_layout = QtWidgets.QVBoxLayout(mode_group)
            self.auto = QtWidgets.QRadioButton(
                "Automatic setup (recommended)", mode_group
            )
            self.existing = QtWidgets.QRadioButton(
                "Use an existing CMTK installation", mode_group
            )
            self.auto.setChecked(True)
            mode_layout.addWidget(self.auto)
            mode_layout.addWidget(self.existing)
            select_layout.addWidget(mode_group)

            if manager.platform_name() == "windows":
                note_text = (
                    "Automatic setup may enable WSL, request Windows administrator "
                    "approval, download and verify an official Ubuntu environment, "
                    "and require a restart. CMTK is installed only in a separately "
                    "named MADI3D-CMTK distribution; personal WSL distributions "
                    "are not modified. WSL 2 requires hardware virtualization to be "
                    "enabled in BIOS/UEFI."
                )
            elif manager.platform_name() == "linux":
                unavailable_reason = _native_setup_unavailable_reason(manager)
                if unavailable_reason:
                    note_text = (
                        unavailable_reason
                        + " Choose an existing CMTK installation."
                    )
                    self.auto.setEnabled(False)
                    self.existing.setChecked(True)
                else:
                    note_text = (
                        "On native Ubuntu/Debian, automatic setup installs the system "
                        "cmtk package from configured repositories and may request "
                        "administrator authentication. MADI3D does not enable additional "
                        "repositories automatically."
                    )
            else:
                note_text = (
                    "Automatic package installation is unavailable on this "
                    "platform. Choose an existing CMTK installation."
                )
                self.auto.setEnabled(False)
                self.existing.setChecked(True)
            note = QtWidgets.QLabel(note_text)
            note.setWordWrap(True)
            select_layout.addWidget(note)
            select_layout.addStretch(1)
            self.stack.addWidget(select_page)

            progress_page = QtWidgets.QWidget(self)
            progress_layout = QtWidgets.QVBoxLayout(progress_page)
            self.stage_label = QtWidgets.QLabel(
                "Checking CMTK...", progress_page
            )
            self.stage_label.setWordWrap(True)
            progress_layout.addWidget(self.stage_label)
            self.progress = QtWidgets.QProgressBar(progress_page)
            self.progress.setRange(0, 0)
            progress_layout.addWidget(self.progress)
            self.log_view = QtWidgets.QPlainTextEdit(progress_page)
            self.log_view.setReadOnly(True)
            self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.log_view.setMinimumHeight(280)
            progress_layout.addWidget(self.log_view, 1)
            self.stack.addWidget(progress_page)

            self.buttons = QtWidgets.QDialogButtonBox(self)
            self.cancel_button = self.buttons.addButton(
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            self.action_button = self.buttons.addButton(
                "Continue", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
            )
            self.cancel_button.clicked.connect(self._cancel_or_reject)
            self.action_button.clicked.connect(self._action)
            root.addWidget(self.buttons)

            if self._force_setup:
                self.stack.setCurrentIndex(0)
                self.action_button.setText("Install / Reinstall")
                self.action_button.setEnabled(True)
            else:
                self.stack.setCurrentIndex(1)
                self.action_button.setEnabled(False)
                QtCore.QTimer.singleShot(0, self._start_detection)

        def _append_log(self, text):
            text = str(text or "").rstrip()
            if not text:
                return
            self.log_view.appendPlainText(text)
            bar = self.log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

        def _set_stage(self, text):
            self.stage_label.setText(str(text))
            self._append_log("\n== " + str(text) + " ==")

        def _is_cancel_requested(self):
            return self._cancel_requested_flag

        def _cancel_or_reject(self):
            if not self._running:
                self.reject()
                return
            self._cancel_requested_flag = True
            self.cancel_button.setEnabled(False)
            self.stage_label.setText(
                "Stopping safely; a protected package step may need to finish..."
            )
            self._append_log(
                "\nCancellation requested. MADI3D will not interrupt an active "
                "dpkg/apt installation step."
            )

        def _action(self):
            if self._running:
                return
            if self.restart_message:
                self.reject()
                return
            if self.backend is not None:
                self.accept()
                return
            if self.stack.currentIndex() == 1:
                self.stack.setCurrentIndex(0)
                self.action_button.setText("Continue")
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(True)
                return

            if self.existing.isChecked():
                try:
                    candidate = _choose_existing_backend(manager, self)
                except CMTKSetupCancelled:
                    return
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self, "CMTK setup", str(exc)
                    )
                    return
                self._start_worker(
                    "existing", candidate=candidate, clear_log=True
                )
                return

            self._start_automatic(clear_log=True)

        def _start_detection(self):
            self._start_worker("detect", clear_log=True)

        def _start_automatic(self, clear_log=False):
            if self._running or self.backend is not None:
                return
            try:
                manager.update_config(
                    setup_mode="auto", setup_pending=True,
                    readiness_state="unavailable",
                    readiness_summary="CMTK setup/repair is in progress or pending.",
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "CMTK setup", str(exc)
                )
                self.stack.setCurrentIndex(0)
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(True)
                return
            self._start_worker(
                "automatic", clear_log=clear_log, reinstall=self._reinstall_requested
            )

        def _start_worker(
            self, mode, *, candidate=None, clear_log=False, reinstall=False
        ):
            if self._running:
                return
            self.restart_message = ""
            self._terminal_kind = None
            self._terminal_payload = None
            self._cancel_requested_flag = False
            self._running = True
            self._active_mode = str(mode)
            self.stack.setCurrentIndex(1)
            if clear_log:
                self.log_view.clear()
            self.progress.setRange(0, 0)
            self.action_button.setEnabled(False)
            self.cancel_button.setText("Cancel")
            self.cancel_button.setEnabled(True)

            labels = {
                "detect": "Checking for an existing CMTK installation...",
                "existing": "Validating the selected CMTK installation...",
                "automatic": (
                    "Starting CMTK reinstall/repair..."
                    if reinstall else "Starting automatic CMTK setup..."
                ),
            }
            self.stage_label.setText(labels.get(mode, "Working..."))

            thread = QtCore.QThread(self)
            worker = Worker(
                mode, self._is_cancel_requested, candidate=candidate,
                reinstall=reinstall,
            )
            worker.moveToThread(thread)
            self._thread = thread
            self._worker = worker

            thread.started.connect(worker.run)
            worker.stage.connect(self._set_stage)
            worker.log.connect(self._append_log)
            worker.completed.connect(self._worker_completed)
            worker.not_found.connect(self._worker_not_found)
            worker.failed.connect(self._worker_failed)
            worker.restart_required.connect(
                self._worker_restart_required
            )
            worker.cancelled.connect(self._worker_cancelled)
            worker.completed.connect(thread.quit)
            worker.not_found.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.restart_required.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(self._thread_finished)
            thread.finished.connect(thread.deleteLater)
            thread.start()

        def _worker_completed(self, backend):
            self._terminal_kind = self._active_mode + "_complete"
            self._terminal_payload = backend
            self.stage_label.setText("Finishing CMTK check...")

        def _worker_not_found(self):
            self._terminal_kind = "detect_missing"
            self._terminal_payload = None

        def _worker_failed(self, message):
            self._terminal_kind = "failed"
            self._terminal_payload = str(message)
            self.stage_label.setText("Finishing failed setup step...")

        def _worker_restart_required(self, message):
            self._terminal_kind = "restart"
            self._terminal_payload = str(message)
            self.stage_label.setText("Finishing Windows setup step...")

        def _worker_cancelled(self, message):
            self._terminal_kind = "cancelled"
            self._terminal_payload = str(message)

        def _clear_pending_after_stop(self, active_mode):
            if active_mode != "automatic":
                return
            try:
                manager.update_config(
                    setup_pending=None, setup_mode=None
                )
            except Exception as exc:
                self._append_log(
                    "\nWARNING: Could not clear pending setup state: "
                    + str(exc)
                )

        def _thread_finished(self):
            self._running = False
            kind = self._terminal_kind
            payload = self._terminal_payload
            active_mode = self._active_mode
            self._active_mode = ""
            self._thread = None
            self._worker = None

            if kind == "detect_complete":
                self.backend = payload
                self.accept()
                return

            if kind == "detect_missing":
                self.detected_status.setText(
                    manager.last_status.diagnostic_text()
                    or "CMTK is not configured."
                )
                if self._resume_auto:
                    self._resume_auto = False
                    QtCore.QTimer.singleShot(
                        0, lambda: self._start_automatic(clear_log=False)
                    )
                    return
                self.stack.setCurrentIndex(0)
                self.action_button.setText("Continue")
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(True)
                return

            if kind in ("automatic_complete", "existing_complete"):
                self.backend = payload
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.stage_label.setText("CMTK is ready")
                self._append_log(
                    "\nSetup and validation completed successfully."
                )
                self._append_log(manager.last_status.diagnostic_text())
                self.action_button.setText("Continue")
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                return

            if kind == "restart":
                self.restart_message = str(
                    payload or "Windows restart required."
                )
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.stage_label.setText("Windows restart required")
                self._append_log("\n" + self.restart_message)
                self.action_button.setText("Close")
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                return

            if kind == "cancelled":
                self._clear_pending_after_stop(active_mode)
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
                self.stage_label.setText("CMTK setup cancelled")
                self._append_log(
                    "\n" + str(payload or "CMTK setup was cancelled.")
                )
                self.action_button.setText("Back")
                self.action_button.setEnabled(True)
                self.cancel_button.setEnabled(True)
                return

            message = str(
                payload or "CMTK setup stopped unexpectedly."
            )
            self._clear_pending_after_stop(active_mode)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.stage_label.setText("CMTK setup did not complete")
            self._append_log("\nERROR: " + message)
            self.action_button.setText("Back")
            self.action_button.setEnabled(True)
            self.cancel_button.setEnabled(True)

        def reject(self):
            if self._running:
                self._cancel_or_reject()
                return
            super().reject()

        def closeEvent(self, event):
            if self._running:
                self._cancel_or_reject()
                event.ignore()
                return
            super().closeEvent(event)

    dialog = SetupDialog(parent)
    result = dialog.exec()
    if dialog.restart_message:
        raise CMTKSetupPending(dialog.restart_message)
    if result != QtWidgets.QDialog.DialogCode.Accepted or dialog.backend is None:
        raise CMTKSetupCancelled()
    return dialog.backend


def ensure_cmtk_ready(
    tools=(), *, parent=None, interactive=True, force_setup=False
):
    """Return CMTK without re-probing a persisted-ready backend on normal use."""
    manager = get_cmtk_manager()
    tools = tuple(dict.fromkeys(str(v) for v in tools if str(v).strip()))

    if not force_setup:
        if manager.backend is not None:
            return manager.backend
        backend = manager.restore_persisted_backend()
        if backend is not None:
            return backend

    if not interactive:
        backend = manager.detect(tools)
        if backend is None:
            raise CMTKUnavailableError(manager.last_status.diagnostic_text())
        return backend

    return _setup_dialog(
        manager, tools or CMTK_CORE_TOOLS, parent, force_setup=bool(force_setup)
    )


def install_or_reinstall_cmtk(*, parent=None, tools=CMTK_CORE_TOOLS):
    """Explicit dependency-menu entry point; always opens setup/repair UI."""
    return ensure_cmtk_ready(
        tools, parent=parent, interactive=True, force_setup=True
    )


__all__ = ["ensure_cmtk_ready", "install_or_reinstall_cmtk"]
