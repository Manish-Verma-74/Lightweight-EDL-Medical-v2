"""Efficiency measurements: parameters, MACs, checkpoint size and latency.

MAC convention (frozen): multiply-accumulates of Conv2d and Linear layers only, counted with forward
hooks on one (1, 3, 224, 224) input. BatchNorm, activations, pooling and element-wise operations are
ignored. This is our own pinned counter, tested on layers with known sizes.
"""
import os
import statistics
import time


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def count_macs(model, input_shape=(1, 3, 224, 224)) -> int:
    import torch
    import torch.nn as nn

    macs = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs[0] += int(out.numel() * k)

    def linear_hook(m, inp, out):
        macs[0] += int(out.numel() * m.in_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in hooks:
        h.remove()
    return macs[0]


def checkpoint_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def _summary(times_ms):
    q = statistics.quantiles(times_ms, n=4) if len(times_ms) >= 2 else [times_ms[0]] * 3
    return {"median_ms": statistics.median(times_ms), "q1_ms": q[0], "q3_ms": q[2], "iqr_ms": q[2] - q[0]}


def measure_latency(fn, device: str, warmup: int = 50, iters: int = 200) -> dict:
    """Time fn() with synchronization on CUDA. fn must contain only the measured computation."""
    import torch
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _summary(times)


def benchmark_model(model, framework, device: str, input_shape=(1, 3, 224, 224), warmup=50, iters=200) -> dict:
    """Forward-only and full-inference-path latency (forward + probabilities + native uncertainty)."""
    import torch
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    def forward_only():
        with torch.no_grad():
            model(x)

    def full_path():
        with torch.no_grad():
            framework.predict(model(x))

    return {"forward": measure_latency(forward_only, device, warmup, iters),
            "full_inference": measure_latency(full_path, device, warmup, iters)}
