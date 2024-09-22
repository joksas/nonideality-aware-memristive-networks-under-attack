from enum import Enum, auto
from typing import Optional

import torch

from .nonidealities import LinearityPreservingNonideality, Nonideality


class MappingRule(Enum):
    """Mapping rule for converting synaptic weights to conductances."""

    LOWEST_CONDUCTANCE = auto()
    AVERAGE_CONDUCTANCE = auto()


def net_currents_to_outputs(
    net_currents: torch.Tensor, k_V: float, max_weight: float, G_off: float, G_on: float
) -> torch.Tensor:
    """Convert net output currents of a dot-product engine onto synaptic layer inputs.

    Args:
        net_currents: Net output currents of shape `p x n`
        k_V: Voltage scaling factor.
        max_weight: Assumed maximum weight.
        G_off: Memristor conductance in OFF state.
        G_on: Memristor conductance in ON state.

    Returns:
        Outputs of shape `p x n` of a synaptic layer implemented using memristive crossbars.
    """
    k_G = _k_G(max_weight, G_off, G_on)
    k_I = _k_I(k_V, k_G)
    outputs = net_currents / k_I

    return outputs


def _k_G(max_weight: float, G_off: float, G_on: float) -> float:
    """Compute conductance scaling factor.

    Args:
        max_weight: Assumed maximum weight.
        G_off: Memristor conductance in OFF state.
        G_on: Memristor conductance in ON state.

    Returns:
        Conductance scaling factor.
    """
    torch._assert(G_off < G_on, "G_off must be smaller than G_on")
    torch._assert(max_weight > 0, "max_weight must be positive")

    return (G_on - G_off) / max_weight


def _k_I(k_V: float, k_G: float) -> float:
    """Compute current scaling factor.

    Args:
        k_V: Voltage scaling factor.
        k_G: Conductance scaling factor.

    Returns:
        Current scaling factor.
    """
    return k_V * k_G


def inputs_to_voltages(inputs: torch.Tensor, k_V: float) -> torch.Tensor:
    """Convert synaptic layer inputs to voltages.

    Args:
        inputs: Synaptic inputs.
        k_V: Voltage scaling factor.

    Returns:
        Voltages applied to the crossbar array.
    """
    return inputs * k_V


def weights_to_conductances(
    weights: torch.Tensor, G_off: float, G_on: float, mapping_rule: MappingRule
) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    """Convert synaptic layer weights to conductances intended to be used in a differential scheme.

    Args:
        weights: Synaptic weights.
        G_off: Memristor conductance in OFF state.
        G_on: Memristor conductance in ON state.
        mapping_rule: Mapping rule for converting synaptic weights to conductances.

    Returns:
        (G_pos, G_neg): Conductances of the positive and negative pairs.
        max_weight: Assumed maximum weight.
    """
    max_weight = float(torch.max(torch.abs(weights)))
    k_G = _k_G(float(max_weight), G_off, G_on)
    G_eff = k_G * weights

    if mapping_rule == MappingRule.LOWEST_CONDUCTANCE:
        # We implement the pairs by choosing the lowest possible conductances.
        G_pos = torch.clamp(G_eff, min=0.0) + G_off
        G_neg = -torch.clamp(G_eff, max=0.0) + G_off
    elif mapping_rule == MappingRule.AVERAGE_CONDUCTANCE:
        # We map the pairs symmetrically around `G_avg`.
        G_avg = (G_off + G_on) / 2
        G_pos = G_avg + 0.5 * G_eff
        G_neg = G_avg - 0.5 * G_eff

    return (G_pos, G_neg), max_weight


class PlacementScheme(Enum):
    """Scheme for choosing the placement of conductances onto the crossbar array."""

    """Place positive and negative conductances in neighbouring bit lines."""
    INTERLEAVED_BIT_LINES = auto()


class Crossbar:
    def __init__(
        self,
        num_word_lines: int,
        num_bit_lines: int,
        nonidealities: list[Nonideality],
        placement_scheme: PlacementScheme = PlacementScheme.INTERLEAVED_BIT_LINES,
    ) -> None:
        """Initialize a crossbar.

        Args:
            num_word_lines: Number of word lines.
            num_bit_lines: Number of bit lines.
        """
        self._num_word_lines = num_word_lines
        self._num_bit_lines = num_bit_lines
        self._placement_scheme = placement_scheme

        self._linearity_presering_nonideality: Optional[
            LinearityPreservingNonideality
        ] = None
        for nonideality in nonidealities:
            if isinstance(nonideality, LinearityPreservingNonideality):
                torch._assert(
                    self._linearity_presering_nonideality is None,
                    "at most one linearity-preserving nonideality is supported",
                )
                self._linearity_presering_nonideality = nonideality

        self.G = torch.zeros(num_word_lines, num_bit_lines)

    @staticmethod
    def device_currents_to_output_currents(
        device_currents: torch.Tensor, interconnect_resistance: float = 0.0
    ) -> torch.Tensor:
        """Compute output currents from device currents.

        Args:
            device_currents: Device currents of shape `* x num_word_lines x num_bit_lines`.

        Returns:
            Output currents of shape `* x num_bit_lines`.
        """
        torch._assert(
            interconnect_resistance == 0.0,
            "currently only zero interconnect resistance is supported",
        )

        return torch.sum(device_currents, dim=-2)

    def place_conductances(self, G_pos: torch.Tensor, G_neg: torch.Tensor) -> None:
        """Place conductances onto the crossbar array.

        Args:
            G_pos: Positive conductances of shape `num_outputs x num_inputs`.
            G_neg: Negative conductances of shape `num_outputs x num_inputs`.
        """
        torch._assert(
            G_pos.shape == G_neg.shape, "G_pos and G_neg must have the same shape"
        )
        num_outputs, num_inputs = tuple(G_pos.shape)

        if self._placement_scheme == PlacementScheme.INTERLEAVED_BIT_LINES:
            torch._assert(
                num_inputs <= self._num_word_lines,
                "the number of inputs must not exceed the number of word lines",
            )
            torch._assert(
                2 * num_outputs <= self._num_bit_lines,
                "twice the number of outputs must not exceed the number of bit lines",
            )
            self.G[:num_inputs, : 2 * num_outputs : 2] = G_pos.T
            self.G[:num_inputs, 1 : 2 * num_outputs : 2] = G_neg.T
        else:
            raise NotImplementedError

    def compute_device_currents(self, voltages: torch.Tensor) -> torch.Tensor:
        """Compute device currents from voltages.

        Args:
            voltages: Voltages of shape `* x num_word_lines`.

        Returns:
            Device currents of shape `* x num_word_lines x num_bit_lines`.
        """
        torch._assert(
            voltages.shape[-1] == self._num_word_lines,
            "the last dimension of voltages must be equal to the number of word lines",
        )

        G = self.G

        if self._linearity_presering_nonideality is not None:
            G = self._linearity_presering_nonideality.disturb_conductances(G)

        device_currents = torch.einsum("ij,jk->ijk", voltages, G.to(voltages.device))

        return device_currents

    def output_currents_to_net_currents(
        self, output_currents: torch.Tensor
    ) -> torch.Tensor:
        """Compute net currents from output currents."""
        if self._placement_scheme == PlacementScheme.INTERLEAVED_BIT_LINES:
            return output_currents[:, ::2] - output_currents[:, 1::2]

        raise NotImplementedError
