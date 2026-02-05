"""Quick AMD GPU validation script for openpi.

Tests that the core PyTorch and JAX operations used by openpi work correctly
on AMD ROCm GPUs. Useful for verifying a new AMD environment is set up properly.
"""

from __future__ import annotations

import sys
import time


def test_pytorch_gpu():
    """Test PyTorch operations on AMD GPU."""
    import torch

    if not torch.cuda.is_available():
        print("SKIP: No GPU available for PyTorch")
        return False

    device = torch.device("cuda:0")
    print(f"PyTorch device: {torch.cuda.get_device_name(0) or 'AMD GPU'}")
    print(f"HIP version: {torch.version.hip}")

    # Matmul warmup + benchmark
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    # Warmup
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    ms = (time.time() - start) / 10 * 1000
    print(f"  bf16 matmul 2048x2048: {ms:.2f}ms")

    # Conv2d (SigLIP-style patch embedding)
    conv = torch.nn.Conv2d(3, 1152, 14, stride=14).to(device).to(torch.bfloat16)
    x = torch.randn(4, 3, 224, 224, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        conv(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        conv(x)
    torch.cuda.synchronize()
    ms = (time.time() - start) / 10 * 1000
    print(f"  SigLIP patch embed (bs=4): {ms:.2f}ms")

    # Transformer-style forward+backward
    model = torch.nn.TransformerEncoderLayer(d_model=2048, nhead=16, batch_first=True).to(device).to(torch.bfloat16)
    x = torch.randn(4, 256, 2048, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        model(x).sum().backward()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        y = model(x)
        y.sum().backward()
    torch.cuda.synchronize()
    ms = (time.time() - start) / 5 * 1000
    print(f"  Transformer layer fwd+bwd (bs=4, seq=256, d=2048): {ms:.2f}ms")

    print(f"  GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")
    return True


def test_jax_gpu():
    """Test JAX operations on AMD ROCm GPU."""
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu":
        print("SKIP: JAX not using GPU backend")
        return False

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].platform}")

    # Matmul
    a = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
    b = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
    c = jnp.matmul(a, b)
    c.block_until_ready()
    start = time.time()
    for _ in range(10):
        c = jnp.matmul(a, b)
    c.block_until_ready()
    ms = (time.time() - start) / 10 * 1000
    print(f"  bf16 matmul 2048x2048: {ms:.2f}ms")

    return True


def test_openpi_imports():
    """Test that openpi modules import correctly."""
    sys.path.insert(0, "/home/openpi/src")

    from openpi.shared import gpu_utils
    print(f"GPU vendor: {gpu_utils.detect_gpu_vendor().value}")
    print(f"DDP backend: {gpu_utils.get_ddp_backend()}")

    from openpi.training import config
    c = config.get_config("debug")
    print(f"Config loaded: {c.name} (model: {c.model.paligemma_variant})")

    return True


def main():
    print("=" * 60)
    print("OpenPI AMD GPU Validation")
    print("=" * 60)

    results = {}

    print("\n--- openpi imports ---")
    results["imports"] = test_openpi_imports()

    print("\n--- PyTorch GPU ---")
    results["pytorch"] = test_pytorch_gpu()

    print("\n--- JAX GPU ---")
    results["jax"] = test_jax_gpu()

    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
