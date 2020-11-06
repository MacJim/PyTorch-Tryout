"""
Tests Torch's normalization functions.
"""

import random

import torch
import torch.nn.functional as F


# MARK: - L* normalization
def test_L1_normalization_1():
    x1 = torch.tensor([1, 1, 1, 5], dtype=torch.float32)
    print(f"x1: {x1}")    # tensor([1., 1., 1., 5.])
    print(f"L1 normalized: {F.normalize(x1, p=1, dim=0)}")    # tensor([0.1250, 0.1250, 0.1250, 0.6250])


def test_L2_normalization_1():
    x1 = torch.tensor(list(range(10)), dtype=torch.float32)
    print(f"x1: {x1}")    # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    print(f"L2 normalized: {F.normalize(x1, p=2, dim=0)}")    # tensor([0.0000, 0.0592, 0.1185, 0.1777, 0.2369, 0.2962, 0.3554, 0.4146, 0.4739, 0.5331])

    x2 = torch.tensor([4] * 8, dtype=torch.float32)
    x2 = x2.view(4, -1)
    print(f"x2: {x2}")    # ([[4., 4.], [4., 4.], [4., 4.], [4., 4.]])
    print(f"L2 normalized dim=0: {F.normalize(x2, p=2, dim=0)}")    # tensor([[0.5000, 0.5000], [0.5000, 0.5000], [0.5000, 0.5000], [0.5000, 0.5000]]) That is, normalize [4 4 4 4]: 1 / 2 = 4 / sqrt(16 * 4)
    print(f"L2 normalized dim=1: {F.normalize(x2, p=2, dim=1)}")    # tensor([[0.7071, 0.7071], [0.7071, 0.7071], [0.7071, 0.7071], [0.7071, 0.7071]]) That is, normalize [4 4]: sqrt(2) / 2 = 4 / (4 * sqrt(2)) = 4 / sqrt(16 + 16).


def test_L2_normalization_2():
    """
    Can only normalize torch tensors.
    Integer tensors are not supported.
    """
    x1 = torch.tensor(list(range(10)))
    print(f"x1: {x1}")    # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"L2 normalized: {F.normalize(x1, p=1, dim=0)}")    # RuntimeError: Can only calculate the mean of floating types. Got Long instead.


# MARK: - Main
if (__name__ == "__main__"):
    test_L1_normalization_1()
    # test_L2_normalization_1()
    # test_L2_normalization_2()

    # test_batch_norm_1()
