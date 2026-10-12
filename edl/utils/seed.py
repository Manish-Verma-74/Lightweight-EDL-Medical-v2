import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and CUDA. GPU kernels may still be non-deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pure-NumPy tests
        pass


def get_rng_states() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def set_rng_states(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Make DataLoader workers deterministic given the loader's generator."""
    try:
        import torch
        seed = torch.initial_seed() % (2 ** 32)
    except ImportError:
        seed = worker_id
    random.seed(seed)
    np.random.seed(seed)
