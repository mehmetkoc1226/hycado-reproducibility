"""Small, dependency-free utilities shared by the clean pilot."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace one report atomically; never write NaN as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def system_info(include_torch=False):
    info = {"created_utc": utc_now(), "os": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2)}
    try:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            total = status.total_phys
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["system_ram_gib"] = round(total / 2**30, 2)
    except (OSError, ValueError, AttributeError):
        info["system_ram_gib"] = None
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        info["nvidia_smi"] = run.stdout.strip() if run.returncode == 0 else run.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "nvidia-smi bulunamadi veya cevap vermedi"
    if include_torch:
        try:
            import importlib.metadata
            import torch
            info["torch"] = torch.__version__
            info["torchvision"] = importlib.metadata.version("torchvision")
            info["ultralytics"] = importlib.metadata.version("ultralytics")
            info["cuda_available"] = torch.cuda.is_available()
            info["torch_cuda_runtime"] = torch.version.cuda
            if info["cuda_available"]:
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            info["torch_import_error"] = str(exc)
    return info
