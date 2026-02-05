"""GPU utility functions for cross-vendor (NVIDIA/AMD) compatibility.

PyTorch's ROCm backend exposes AMD GPUs through the torch.cuda.* API, so most
torch.cuda calls work on both NVIDIA and AMD without changes. This module provides
helpers for the cases where vendor-specific behavior is needed (e.g., GPU info
queries, memory config, backend-specific optimizations).
"""

import enum
import logging
import os
import subprocess


class GpuVendor(enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    NONE = "none"


def detect_gpu_vendor() -> GpuVendor:
    """Detect the GPU vendor available on this system."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GpuVendor.NONE
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return GpuVendor.AMD
        if torch.version.cuda is not None:
            return GpuVendor.NVIDIA
    except ImportError:
        pass
    return GpuVendor.NONE


def is_amd_gpu() -> bool:
    return detect_gpu_vendor() == GpuVendor.AMD


def is_nvidia_gpu() -> bool:
    return detect_gpu_vendor() == GpuVendor.NVIDIA


def has_gpu() -> bool:
    """Check if any GPU (NVIDIA or AMD) is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_available_no_torch() -> bool:
    """Check if a GPU is available without importing torch.

    Uses vendor-specific CLI tools (nvidia-smi or rocm-smi) to detect GPUs.
    This is useful in test fixtures that need to check GPU availability before
    torch is configured.
    """
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try AMD
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def configure_memory_optimizations(world_size: int = 1) -> None:
    """Apply vendor-appropriate memory optimizations for large-scale training.

    For NVIDIA: enables cuDNN benchmark, TF32 matmul, and CUDA allocator config.
    For AMD: enables MIOpen tuning and HIP memory allocator config.
    """
    import torch

    vendor = detect_gpu_vendor()

    if world_size >= 8:
        if vendor == GpuVendor.NVIDIA:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
            logging.info("Enabled NVIDIA memory optimizations for 8+ GPU training")
        elif vendor == GpuVendor.AMD:
            # MIOpen (AMD's equivalent of cuDNN) auto-tuning
            os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
            # HIP memory allocator configuration
            os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
            # Enable TF32-equivalent on AMD (matrix core operations)
            torch.backends.cuda.matmul.allow_tf32 = True
            logging.info("Enabled AMD memory optimizations for 8+ GPU training")


def get_ddp_backend() -> str:
    """Get the appropriate DDP backend for the current GPU vendor.

    Returns "nccl" for NVIDIA (uses NCCL), "nccl" for AMD (uses RCCL which
    PyTorch exposes as the "nccl" backend), or "gloo" if no GPU is available.
    """
    import torch

    if torch.cuda.is_available():
        # ROCm's RCCL is exposed as the "nccl" backend in PyTorch
        return "nccl"
    return "gloo"


def get_oom_hint() -> str:
    """Get a vendor-appropriate hint for out-of-memory errors."""
    vendor = detect_gpu_vendor()
    if vendor == GpuVendor.AMD:
        return "Out of memory while loading checkpoint. Try setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"
    return "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
