#!/usr/bin/env python3
"""Benchmark Pi0 policy inference (sample_actions) on AMD GPU.

Measures end-to-end latency including image encoding, language encoding,
prefill, and denoising steps. This is a clean baseline -- no optimizations
are applied here. The agent's job is to optimize the model code and
environment to reduce P50 latency.

IMMUTABLE: Do NOT change num_steps, batch_size, iterations, or warmup.
"""
import os
import sys

sys.path.insert(0, "/sgl-workspace/openpi/src")

import time
import numpy as np
import torch

NUM_STEPS = 10
BATCH_SIZE = 1
WARMUP = 20
ITERATIONS = 100


class SimpleObservation:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_observation(device):
    images = {
        "base_0_rgb": torch.rand(BATCH_SIZE, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        "left_wrist_0_rgb": torch.rand(BATCH_SIZE, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1,
        "right_wrist_0_rgb": torch.zeros(BATCH_SIZE, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(BATCH_SIZE, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(BATCH_SIZE, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device),
    }
    return SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=torch.randn(BATCH_SIZE, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 256000, (BATCH_SIZE, 20), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(BATCH_SIZE, 20, dtype=torch.bool, device=device),
        token_ar_mask=torch.ones(BATCH_SIZE, 20, dtype=torch.int32, device=device),
        token_loss_mask=torch.zeros(BATCH_SIZE, 20, dtype=torch.bool, device=device),
    )


def main():
    print("=" * 70)
    print("PI0 POLICY INFERENCE BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda:7")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    from dataclasses import dataclass
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    @dataclass
    class Pi0Config:
        action_dim: int = 32
        action_horizon: int = 10
        max_token_len: int = 48
        dtype: str = "bfloat16"
        paligemma_variant: str = "gemma_2b"
        action_expert_variant: str = "gemma_300m"
        pi05: bool = False

    print("\nLoading model...")
    model = PI0Pytorch(Pi0Config())
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Parameters: {params_b:.2f}B")

    observation = create_observation(device)

    print(f"\nnum_steps={NUM_STEPS}  batch_size={BATCH_SIZE}  warmup={WARMUP}  iterations={ITERATIONS}")
    print("-" * 70)

    print("Warmup...")
    for _ in range(WARMUP):
        with torch.no_grad():
            model.sample_actions(device, observation, num_steps=NUM_STEPS)
    torch.cuda.synchronize()

    if os.environ.get("PROFILE", "0") == "1":
        from torch.profiler import profile, ProfilerActivity

        trace_dir = os.environ.get("PROFILE_DIR", "/workspace/traces")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, "policy_inference.json")
        print("\nProfiling (1 iteration)...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_flops=True,
        ) as prof:
            with torch.no_grad():
                model.sample_actions(device, observation, num_steps=NUM_STEPS)
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        print(f"Trace: {trace_path} ({os.path.getsize(trace_path) / 1024 / 1024:.1f} MB)")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    print("Benchmarking...")
    latencies = []
    for i in range(ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=NUM_STEPS)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  iter {i+1:3d}/{ITERATIONS}: {lat:.1f} ms")

    print(f"\n{'=' * 70}")
    print(f"RESULTS  (num_steps={NUM_STEPS}, batch_size={BATCH_SIZE})")
    print("=" * 70)
    print(f"Mean: {np.mean(latencies):.1f} ms")
    print(f"Std:  {np.std(latencies):.1f} ms")
    print(f"Min:  {np.min(latencies):.1f} ms")
    print(f"Max:  {np.max(latencies):.1f} ms")
    print(f"P50:  {np.percentile(latencies, 50):.1f} ms")
    print(f"P95:  {np.percentile(latencies, 95):.1f} ms")
    print(f"P99:  {np.percentile(latencies, 99):.1f} ms")
    print(f"Actions shape: {tuple(actions.shape)}")
    print(f"Throughput:    {1000 / np.mean(latencies):.2f} Hz")
    print(f"Memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")
    print("=" * 70)


if __name__ == "__main__":
    main()
