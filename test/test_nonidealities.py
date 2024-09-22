import pytest
import torch
from torch.testing import assert_close

from memnet import mapping, nonidealities, utils

linearity_preserving_testdata = [
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nonidealities.StuckAt(2.5, 0.0),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ),
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nonidealities.StuckAt(2.5, 1.0),
        torch.tensor([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]]),
    ),
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nonidealities.StuckAtGOff(1.0, 0.0),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ),
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        nonidealities.StuckAtGOff(1.0, 1.0),
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ),
]


@pytest.mark.parametrize(
    "G,nonideality,G_disturbed_expected", linearity_preserving_testdata
)
def test_linearity_preserving(
    G, nonideality: nonidealities.LinearityPreservingNonideality, G_disturbed_expected
):
    G_disturbed = nonideality.disturb_conductances(G)
    assert_close(G_disturbed, G_disturbed_expected)


label_testdata = [
    (
        nonidealities.StuckAt(2.5, 0.0),
        False,
        "Stuck={value=2.5,p=0}",
    ),
    (
        nonidealities.StuckAt(2.5, 1.0),
        False,
        "Stuck={value=2.5,p=1}",
    ),
    (
        nonidealities.StuckAt(2.5, 0.5),
        False,
        "Stuck={value=2.5,p=0.5}",
    ),
    (
        nonidealities.StuckAt(2.5, 1 / 3),
        False,
        "Stuck={value=2.5,p=0.333}",
    ),
    (
        nonidealities.StuckAtGOff(1.0, 0.0),
        False,
        "StuckOff={G_off=1,p=0}",
    ),
    (
        nonidealities.StuckAtGOff(1.0, 1.0),
        False,
        "StuckOff={G_off=1,p=1}",
    ),
    (
        nonidealities.StuckAtGOff(1.0, 0.5),
        False,
        "StuckOff={G_off=1,p=0.5}",
    ),
    (
        nonidealities.StuckAtGOff(1.0, 1 / 3),
        True,
        "StuckOff={G_off=1,p=0.333}_initialized",
    ),
]
