import pytest
import torch
from torch.testing import assert_close

from memnet import layers

combined_weights_shape_testdata = [
    (
        3,
        7,
        (7, 4),
    ),
    (
        1,
        1,
        (1, 2),
    ),
]


@pytest.mark.parametrize(
    "in_features,out_features,shape", combined_weights_shape_testdata
)
def test_combined_weights_shape(
    in_features: int, out_features: int, shape: tuple[int, int]
):
    layer = layers.MemristiveLayer(in_features, out_features, [])
    combined_weights = layer.combined_weights()
    assert tuple(combined_weights.shape) == shape


combined_inputs_testdata = [
    (
        3,
        7,
        torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        torch.tensor([[1.0, 4.0, 1.0], [2.0, 5.0, 1.0], [3.0, 6.0, 1.0]]),
    ),
]


@pytest.mark.parametrize(
    "in_features,out_features,inputs,expected", combined_inputs_testdata
)
def test_combined_inputs(
    in_features: int,
    out_features: int,
    inputs: torch.Tensor,
    expected: torch.Tensor,
):
    layer = layers.MemristiveLayer(in_features, out_features, [])
    combined_inputs = layer.combined_inputs(inputs)
    assert_close(combined_inputs, expected)
