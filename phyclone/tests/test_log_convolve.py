import numpy as np
import unittest
from phyclone.tree_utils import conv_log


def numpy_reliant_non_log_conv(child_log_R, prev_log_D_n):
    """ Compute the recursion over D using the numpy.
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = np.convolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max


class Test(unittest.TestCase):

    def test_small_grid_simple_integers(self):
        child_1_no_log = np.array([9, 8, 7, 6, 5, 4, 3, 2, 10])
        child_2_no_log = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])

        child_1 = np.log(child_1_no_log)
        child_2 = np.log(child_2_no_log)
        actual = conv_log(child_1, child_2, np.zeros_like(child_1))

        expected = numpy_reliant_non_log_conv(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_empty_arrays(self):
        child_1_no_log = np.array([])
        child_2_no_log = np.array([])

        child_1 = np.log(child_1_no_log)
        child_2 = np.log(child_2_no_log)
        actual = conv_log(child_1, child_2, np.zeros_like(child_1))

        expected = []

        np.testing.assert_allclose(actual, expected)

    def test_single_element_arrays(self):
        grid_size = 1
        child_1_no_log = np.full(grid_size, 2)
        child_2_no_log = np.full(grid_size, 3)

        child_1 = np.log(child_1_no_log)
        child_2 = np.log(child_2_no_log)
        actual = conv_log(child_1, child_2, np.zeros_like(child_1))

        expected = numpy_reliant_non_log_conv(child_1, child_2)

        np.testing.assert_allclose(actual, expected)

    def test_default_grid_size_prior(self):
        grid_size = 101
        child_1 = np.full(grid_size, -np.log(grid_size))
        child_2 = np.full(grid_size, -np.log(grid_size))
        actual = conv_log(child_1, child_2, np.zeros_like(child_1))

        expected = numpy_reliant_non_log_conv(child_1, child_2)

        np.testing.assert_allclose(actual, expected)