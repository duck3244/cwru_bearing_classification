import numpy as np
import torch

from utils import set_seed


def test_set_seed_numpy_deterministic() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_torch_deterministic() -> None:
    set_seed(7)
    a = torch.randn(3, 4)
    set_seed(7)
    b = torch.randn(3, 4)
    assert torch.equal(a, b)


def test_set_seed_distinct_seeds_differ() -> None:
    set_seed(1)
    a = torch.randn(10)
    set_seed(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)
