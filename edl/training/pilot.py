"""Epoch-cap rules from the protocol (pure functions, easy to test)."""


def compute_epoch_cap(best_epochs, patience: int = 8, margin: int = 15) -> int:
    """E_max = max(best epochs of the pilots) + patience + margin."""
    if not best_epochs:
        raise ValueError("need at least one pilot best epoch")
    return int(max(best_epochs)) + int(patience) + int(margin)


def pilot_needs_extension(stop_reason: str, best_epoch: int, ceiling: int) -> bool:
    """A pilot triggers an extension if it reached the ceiling without early stopping, or if its best
    validation macro-F1 occurred within the last ten epochs of the current ceiling."""
    return stop_reason == "cap" or best_epoch > ceiling - 10


def next_cap(current_cap: int, step: int, ceiling) -> int:
    """Raise the global cap by `step`, never beyond the absolute resource ceiling."""
    new = int(current_cap) + int(step)
    if ceiling is not None:
        new = min(new, int(ceiling))
    return new
