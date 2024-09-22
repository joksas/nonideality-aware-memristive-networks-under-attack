import pytest
import torch
from torch.testing import assert_close

from memnet import mapping, nonidealities

weights_to_conductances_testdata = [
    (
        torch.Tensor(
            [
                [3.75, 2.5, -5.0],
                [-2.5, 0.0, 1.25],
            ]
        ),
        2.0,
        10.0,
        torch.Tensor(
            [
                [8.0, 6.0, 2.0],
                [2.0, 2.0, 4.0],
            ],
        ),
        torch.Tensor(
            [
                [2.0, 2.0, 10.0],
                [6.0, 2.0, 2.0],
            ]
        ),
        5.0,
    ),
    (
        torch.Tensor(
            [
                [4.0],
                [-2.0],
            ]
        ),
        3.0,
        5.0,
        torch.Tensor(
            [
                [5.0],
                [3.0],
            ]
        ),
        torch.Tensor(
            [
                [3.0],
                [4.0],
            ]
        ),
        4.0,
    ),
]


@pytest.mark.parametrize(
    "weights,G_off,G_on,G_pos_exp,G_neg_exp,max_weight_exp",
    weights_to_conductances_testdata,
)
def test_weights_to_conductances(
    weights: torch.Tensor,
    G_off: float,
    G_on: float,
    G_pos_exp: torch.Tensor,
    G_neg_exp: torch.Tensor,
    max_weight_exp: float,
):
    (G_pos, G_neg), max_weight = mapping.weights_to_conductances(
        weights,
        G_off,
        G_on,
        mapping_rule=mapping.MappingRule.LOWEST_CONDUCTANCE,
    )
    assert_close(G_pos, G_pos_exp)
    assert_close(G_neg, G_neg_exp)
    assert_close(max_weight, max_weight_exp)


crossbar_device_currents_to_output_currents_testdata = [
    (
        torch.Tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ]
        ),
        torch.Tensor([3.0, 5.0, 7.0]),
    ),
    (
        torch.Tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                ],
                [
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                ],
            ]
        ),
        torch.Tensor(
            [
                [3.0, 5.0, 7.0],
                [15.0, 17.0, 19.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "device_currents,output_currents_exp",
    crossbar_device_currents_to_output_currents_testdata,
)
def test_crossbar_device_currents_to_output_currents(
    device_currents: torch.Tensor, output_currents_exp: torch.Tensor
):
    currents_shape = device_currents.shape
    crossbar = mapping.Crossbar(currents_shape[-2], currents_shape[-1], [])
    output_currents = crossbar.device_currents_to_output_currents(device_currents)
    assert_close(output_currents, output_currents_exp)


crossbar_place_conductances_testdata = [
    (
        3,
        4,
        torch.Tensor(
            [
                [8.0, 6.0, 2.0],
                [2.0, 2.0, 4.0],
            ],
        ),
        torch.Tensor(
            [
                [2.0, 2.0, 10.0],
                [6.0, 2.0, 2.0],
            ]
        ),
        torch.Tensor(
            [
                [8.0, 2.0, 2.0, 6.0],
                [6.0, 2.0, 2.0, 2.0],
                [2.0, 10.0, 4.0, 2.0],
            ]
        ),
    ),
    (
        3,
        6,
        torch.Tensor(
            [
                [8.0, 6.0, 2.0],
                [2.0, 2.0, 4.0],
            ],
        ),
        torch.Tensor(
            [
                [2.0, 2.0, 10.0],
                [6.0, 2.0, 2.0],
            ]
        ),
        torch.Tensor(
            [
                [8.0, 2.0, 2.0, 6.0, 0.0, 0.0],
                [6.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [2.0, 10.0, 4.0, 2.0, 0.0, 0.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "num_word_lines,num_bit_lines,G_pos,G_neg,G_exp",
    crossbar_place_conductances_testdata,
)
def test_crossbar_place_conductances(
    num_word_lines: int,
    num_bit_lines: int,
    G_pos: torch.Tensor,
    G_neg: torch.Tensor,
    G_exp: torch.Tensor,
):
    crossbar = mapping.Crossbar(num_word_lines, num_bit_lines, [])
    crossbar.place_conductances(G_pos, G_neg)
    assert_close(crossbar.G, G_exp)


crossbar_compute_device_currents_testdata = [
    (
        mapping.Crossbar(3, 2, []),
        torch.Tensor([[1.0, 2.0, 3.0]]),
        torch.Tensor([[5.0, 0.0, 6.0]]),
        torch.Tensor(
            [
                [1.0, 2.0, 3.0],
            ]
        ),
        torch.Tensor(
            [
                [
                    [1.0, 5.0],
                    [4.0, 0.0],
                    [9.0, 18.0],
                ]
            ]
        ),
    ),
    (
        mapping.Crossbar(3, 2, [nonidealities.StuckAt(0.0, 0.0)]),
        torch.Tensor([[1.0, 2.0, 3.0]]),
        torch.Tensor([[5.0, 0.0, 6.0]]),
        torch.Tensor(
            [
                [1.0, 2.0, 3.0],
            ]
        ),
        torch.Tensor(
            [
                [
                    [1.0, 5.0],
                    [4.0, 0.0],
                    [9.0, 18.0],
                ]
            ]
        ),
    ),
    (
        mapping.Crossbar(3, 2, [nonidealities.StuckAt(0.0, 1.0)]),
        torch.Tensor([[1.0, 2.0, 3.0]]),
        torch.Tensor([[5.0, 0.0, 6.0]]),
        torch.Tensor(
            [
                [1.0, 2.0, 3.0],
            ]
        ),
        torch.Tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "crossbar,G_pos,G_neg,voltages,device_currents_exp",
    crossbar_compute_device_currents_testdata,
)
def test_crossbar_compute_device_currents(
    crossbar: mapping.Crossbar,
    G_pos: torch.Tensor,
    G_neg: torch.Tensor,
    voltages: torch.Tensor,
    device_currents_exp: torch.Tensor,
):
    crossbar.place_conductances(G_pos, G_neg)
    device_currents = crossbar.compute_device_currents(voltages)
    assert_close(device_currents, device_currents_exp)
