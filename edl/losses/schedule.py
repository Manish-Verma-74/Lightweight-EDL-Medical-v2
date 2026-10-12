"""KL annealing schedule (no torch needed, so it is easy to test)."""


def annealing_coefficient(epoch: int, annealing_step: int) -> float:
    """KL weight min(1, epoch / annealing_step) with ONE-BASED epochs.

    epoch 1 -> 1/step (0.1 for step 10); epoch == step -> 1.0; later epochs stay at 1.0.
    """
    if epoch < 1:
        raise ValueError(f"epoch must be one-based (>= 1), got {epoch}")
    if annealing_step < 1:
        raise ValueError("annealing_step must be >= 1")
    return min(1.0, float(epoch) / float(annealing_step))
