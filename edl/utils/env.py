"""Runtime metadata and fingerprints for reproducible multi-worker training."""
import hashlib
import platform
import subprocess
import sys
from pathlib import Path


def git_commit(repo_dir=".") -> str:
    """Current commit hash, or 'unknown' outside a Git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def git_is_dirty(repo_dir=".") -> bool:
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, timeout=20)
        return bool(out.stdout.strip())
    except Exception:
        return False


def training_source_fingerprint(repo_dir=".") -> str:
    """Fingerprint training-critical Python sources; evaluation-only edits do not invalidate checkpoints.

    The allowlist deliberately includes the data sampling/transforms, augmentation, model/head, losses,
    training engine/runner, config and seed handling. A config-only epoch-cap commit preserves this hash,
    but a training-affecting code change does not.
    """
    root = Path(repo_dir).resolve()
    rels = []
    for sub in ["edl/training", "edl/models", "edl/losses"]:
        rels.extend(p for p in (root / sub).rglob("*.py") if "__pycache__" not in p.parts)
    rels.extend(root / p for p in [
        "edl/augment.py", "edl/frameworks.py", "edl/config.py", "edl/constants.py", "edl/data/ham10000.py",
        "edl/data/split.py", "edl/utils/seed.py", "edl/utils/hashing.py", "edl/utils/io.py",
        "edl/utils/env.py", "scripts/run_experiments.py",
    ] if (root / p).exists())
    h = hashlib.sha256()
    for p in sorted(set(rels), key=lambda x: x.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def collect_env(repo_dir=".") -> dict:
    """Software and hardware identity, saved with every run."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "git_commit": git_commit(repo_dir), "training_source_fingerprint": training_source_fingerprint(repo_dir)}
    for name in ["numpy", "pandas", "scipy", "sklearn", "PIL", "torch", "torchvision", "yaml"]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[name] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_build"] = torch.version.cuda
        info["gpu_model"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda_available"] = False
        info["gpu_model"] = None
    return info


def check_runtime_environment(cfg: dict, environment_file: str, *, require_exact: bool = True) -> dict:
    """Compare torch, torchvision and scikit-learn to environment/colab_recorded.txt.

    Checks major/minor parity for torch/torchvision (including build suffix separately in returned data)
    and exact scikit-learn version, as the splitter's assignment depends on that version.
    """
    expected = {}
    p = Path(environment_file)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = __import__("re").match(r"^\s*(python|torch|torchvision|scikit-learn)\s+([^\s]+)", line)
            if m: expected[m.group(1)] = m.group(2)
    import sklearn
    import torch
    import torchvision
    actual = {"python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "scikit-learn": sklearn.__version__}
    errors = []
    if "scikit-learn" in expected and actual["scikit-learn"] != expected["scikit-learn"]:
        errors.append(f"scikit-learn expected {expected['scikit-learn']}, found {actual['scikit-learn']}")
    if "python" in expected:
        ev_py = expected["python"].split(".")[:2]
        av_py = actual["python"].split(".")[:2]
        if ev_py != av_py:
            errors.append(f"python major/minor expected {'.'.join(ev_py)}, found {'.'.join(av_py)}")
    for name in ("torch", "torchvision"):
        if name in expected:
            ev = expected[name].split("+", 1)[0].split(".")[:2]
            av = actual[name].split("+", 1)[0].split(".")[:2]
            if ev != av:
                errors.append(f"{name} major/minor expected {'.'.join(ev)}, found {'.'.join(av)}")
    if errors and require_exact:
        raise RuntimeError("runtime environment mismatch; do not mix workers until investigated: " + "; ".join(errors))
    return {"expected": expected, "actual": actual, "mismatches": errors}
