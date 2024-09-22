import math
from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from . import mapping
from .nonidealities import Nonideality


class MemristiveParams:
    def __init__(
        self, G_off: float, G_on: float, k_V: float, nonidealities: list[Nonideality]
    ):
        self.G_off, self.G_on = G_off, G_on
        self.k_V = k_V
        self.nonidealities = nonidealities

    def label(self) -> str:
        """Returns a label for this set of memristive parameters."""
        label_str = f"G_off={self.G_off:.3g}_G_on={self.G_on:.3g}_k_V={self.k_V:.3g}"
        if len(self.nonidealities) > 0:
            nonidealities_label = "+".join([n.label() for n in self.nonidealities])
            label_str += f"_{nonidealities_label}"
        return label_str


class MemristiveLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        memristive_params: Optional[MemristiveParams],
    ):
        super(MemristiveLayer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.memristive_params = memristive_params
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = Parameter(torch.empty((out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Adapted from
        <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py>
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def combined_weights(self) -> torch.Tensor:
        """Returns weights and biases combined into a single tensor."""
        return torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)

    def combined_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns inputs that include a row of ones for the bias."""
        return torch.cat([x, torch.ones(x.shape[0], 1).to(x.device)], dim=1)

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def forward_memristive(self, x: torch.Tensor) -> torch.Tensor:
        assert self.memristive_params is not None

        G_off, G_on, k_V = (
            self.memristive_params.G_off,
            self.memristive_params.G_on,
            self.memristive_params.k_V,
        )
        nonidealities = self.memristive_params.nonidealities

        inputs = self.combined_inputs(x)
        voltages = mapping.inputs_to_voltages(inputs, k_V)

        weights = self.combined_weights()

        (G_pos, G_neg), max_weight = mapping.weights_to_conductances(
            weights, G_off, G_on, mapping.MappingRule.LOWEST_CONDUCTANCE
        )

        crossbar = mapping.Crossbar(
            self.in_features + 1, 2 * self.out_features, nonidealities
        )
        crossbar.place_conductances(G_pos, G_neg)

        device_currents = crossbar.compute_device_currents(voltages)
        output_currents = mapping.Crossbar.device_currents_to_output_currents(
            device_currents
        )
        net_currents = crossbar.output_currents_to_net_currents(output_currents)
        outputs = mapping.net_currents_to_outputs(
            net_currents, k_V, max_weight, G_off, G_on
        )

        if not outputs.requires_grad:
            outputs.requires_grad = True

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.memristive_params is None:
            return self.forward_linear(x)

        return self.forward_memristive(x)
