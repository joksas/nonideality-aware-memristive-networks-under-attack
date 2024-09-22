from abc import ABC, abstractmethod

import torch

from . import utils


class Nonideality(ABC):
    """Physical effect that influences the behavior of memristive devices."""

    def __init__(self) -> None:
        self._parameters = None

    @abstractmethod
    def _label(self) -> str:
        """Returns nonideality label used in directory names, for example."""

    def label(self) -> str:
        label_str = self._label()
        if self._parameters is not None:
            label_str += "_initialized"

        return label_str

    def __eq__(self, other):
        if self is None or other is None:
            if self is None and other is None:
                return True
            return False
        return self.label() == other.label()


class LinearityPreservingNonideality(ABC):
    """Nonideality whose effect can be simulated by disturbing the conductances."""

    @abstractmethod
    def disturb_conductances(self, G: torch.Tensor) -> torch.Tensor:
        """Returns conductances disturbed by the nonideality."""


class StuckAt(Nonideality, LinearityPreservingNonideality):
    """Models a fraction of the devices as stuck in one conductance state."""

    def __init__(self, value: float, probability: float) -> None:
        """
        Args:
            value: Conductance value to set randomly selected devices to.
            probability: Probability that a given device will be set to `val`.
                Probability must be in the [0.0, 1.0] range.
        """
        super().__init__()
        torch._assert(value >= 0.0, "conductance value must be non-negative")
        torch._assert(probability >= 0.0, "probability must be non-negative")
        torch._assert(
            probability <= 1.0, "probability must be less than or equal to 1.0"
        )
        self.value = value
        self.probability = probability

    def _label(self):
        return "Stuck={" f"value={self.value:.3g}," f"p={self.probability:.3g}" "}"

    def disturb_conductances(self, G):
        mask = utils.random_bool_tensor(list(G.size()), self.probability)
        stuck_G = torch.where(mask, self.value, G)

        return stuck_G


class StuckAtGOff(StuckAt):
    """Models a fraction of the devices as stuck in the OFF conductance state."""

    def __init__(self, G_off: float, probability: float) -> None:
        """
        Args:
            G_off: OFF conductance value.
            probability: Probability that a given device will be set to `G_off`.
                Probability must be in the [0.0, 1.0] range.
        """
        super().__init__(G_off, probability)

    def _label(self):
        return "StuckOff={" f"G_off={self.value:.3g}," f"p={self.probability:.3g}" "}"
