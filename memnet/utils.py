import torch


def random_bool_tensor(shape: list[int], prob_true: float) -> torch.Tensor:
    """Return random boolean tensor.

    Args:
        shape: Tensor shape.
        prob_true: Probability that a given entry is going to be True. Probability must be in the
            [0.0, 1.0] range.
    """
    random_float_tensor = torch.rand(shape, dtype=torch.float32)
    return random_float_tensor < prob_true
