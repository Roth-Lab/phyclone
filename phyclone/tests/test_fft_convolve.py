import unittest

import numpy as np

from phyclone.tree.utils import _np_conv_dims
from phyclone.utils.math import fft_convolve_two_children


def run_direct_two_child_convolve(child_1, child_2):
    res_arr = _np_conv_dims(child_1, child_2)
    return res_arr


class Test(unittest.TestCase):

    def __init__(self, method_name: str = ...):
        super().__init__(method_name)

        self.big_grid = 1001
        self.default_grid_size = 101

        self.rng = np.random.default_rng(12345)

    def test_default_grid_size_prior_1_dim(self):
        grid_size = self.default_grid_size
        dim = 1
        child_1 = np.full((dim, grid_size), -np.log(grid_size))
        child_2 = np.full((dim, grid_size), -np.log(grid_size))

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_default_grid_size_prior_4_dim(self):
        grid_size = self.default_grid_size
        dim = 4
        child_1 = np.full((dim, grid_size), -np.log(grid_size))
        child_2 = np.full((dim, grid_size), -np.log(grid_size))

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_random_floats_1_dim_default_gridsize(self):
        grid_size = self.default_grid_size
        dim = 1

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_random_floats_1_dim_big_gridsize(self):
        grid_size = self.big_grid
        dim = 1

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_random_floats_4_dim_big_gridsize(self):
        grid_size = self.big_grid
        dim = 4

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order="C")

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)
