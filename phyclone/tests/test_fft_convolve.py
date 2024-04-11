import numpy as np
import unittest
from phyclone.tree.utils import _conv_two_children_jit
from phyclone.utils.math import fft_convolve_two_children


def run_direct_two_child_convolve(child_1, child_2):
    num_dims = child_1.shape[0]
    res_arr = np.empty_like(child_1)
    _conv_two_children_jit(child_1, child_2, num_dims, res_arr)
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

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_random_floats_1_dim_big_gridsize(self):
        grid_size = self.big_grid
        dim = 1

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_random_floats_4_dim_big_gridsize(self):
        grid_size = self.big_grid
        dim = 4

        child_1 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')
        child_2 = np.log(self.rng.uniform(low=0.1, high=1000, size=(dim, grid_size)), order='C')

        actual = fft_convolve_two_children(child_1, child_2)

        expected = run_direct_two_child_convolve(child_1, child_2)

        np.testing.assert_allclose(actual, expected)
