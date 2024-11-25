import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent 
from torch.utils.data import DataLoader

from .data import Dataset, Subset, transform_loader

saved = False


class AttackType(Enum):
    UNTARGETED = "untargeted"
    TARGETED = "targeted"


class LpNorm(Enum):
    L1 = 1
    L2 = 2
    L_INF = math.inf

    def __get__(self, instance, owner):
        return self.value


class Attack(ABC):
    """Base class for attacks."""

    def __init__(
        self,
        attack_type: AttackType,
        clip_min: float,
        clip_max: float,
    ):
        self._attack_type = attack_type
        self._clip_min = clip_min
        self._clip_max = clip_max

    @abstractmethod
    def _perturb_inputs(
        self, model: torch.nn.Module, x: torch.Tensor, y: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Transforms the input data."""

    @abstractmethod
    def label(self) -> str:
        """Returns the label for the attack."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Returns the label for the attack."""

    def apply(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        subset: Subset,
        target_class: Optional[int] = None,
    ) -> DataLoader:
        """Transforms the input data in the data loader and also returns distances."""
        batches: list[tuple[Any, Any]] = []
        loader = dataset.loader(subset)

        for data, target in loader:
            # Temporary fix because `torchattacks` always uses `y`.
            y = target if self.name() == r"PGD $\ell_2$" else None
            if self._attack_type == AttackType.TARGETED:
                assert target_class is not None
                y = torch.full_like(target, target_class)

            perturbed_data = self._perturb_inputs(model, data, y)
            batches.append((perturbed_data, target))

        examples = [(x, y) for batch in batches for x, y in zip(*batch)]

        return transform_loader(examples, subset)


class FGSM(Attack):
    """Fast Gradient Sign Method (FGSM) attack."""

    def __init__(
        self,
        attack_type: AttackType,
        epsilon: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(attack_type, clip_min, clip_max)
        self._epsilon = epsilon

    def _perturb_inputs(self, model, x, y):
        return fast_gradient_method(
            model,
            x,
            self._epsilon,
            math.inf,
            y=y,
            targeted=(self._attack_type == AttackType.TARGETED),
            clip_min=self._clip_min,
            clip_max=self._clip_max,
        )

    @classmethod
    def name(cls) -> str:
        return "FGSM"

    def label(self) -> str:
        return f"{self.name()}={{epsilon={self._epsilon}}}"


class PGD_L2(Attack):
    """Projected Gradient Descent (PGD) L2 attack."""

    def __init__(
        self,
        attack_type: AttackType,
        epsilon: float,
        epsilon_iter: float,
        num_iterations: int,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(attack_type, clip_min, clip_max)
        self._epsilon = epsilon
        self._num_iterations = num_iterations
        self._epsilon_iter = epsilon_iter 

    def _perturb_inputs(self, model, x, y):
        return projected_gradient_descent(
            model,
            x,
            self._epsilon,
            self._epsilon_iter,
            self._num_iterations,
            2,
            y=y,
            targeted=(self._attack_type == AttackType.TARGETED),
            clip_min=self._clip_min,
            clip_max=self._clip_max,
        )

    @classmethod
    def name(cls) -> str:
        return "PGD $\ell_2$"

    def label(self) -> str:
        return f"{self.name()}={{epsilon={self._epsilon},num_iterations={self._num_iterations},epsilon_iter={self._epsilon_iter}}}"