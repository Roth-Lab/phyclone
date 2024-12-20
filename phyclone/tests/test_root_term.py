import unittest

import numpy as np


class Test(unittest.TestCase):

    def naive_way(self, r, b_term):
        res = 0
        a_term = 1**b_term
        for i in range(1, r + 1):
            r_term = 1 / (1000 ** (i - 1))
            res += a_term * r_term

        if r == 0:
            res = a_term

        return res

    def closed_form(self, r, b_term):
        a_term = 1**b_term

        if r == 0:
            res = a_term
        else:
            r_term_numerator = 1 - (1 / (1000**r))
            r_term_denominator = 1 - (1 / (1000**1))

            res = a_term * (r_term_numerator / r_term_denominator)
        return res

    def overall_prob(self, r, z_term):

        res = 1 / (z_term * (1000 ** (r - 1)))
        return res

    def closed_form_log(self, r, b_term):
        a_term = np.log(1) * b_term
        la = np.log(1)

        if r == 0:
            res = a_term
        else:

            r_term_numerator = np.log(1) - (np.log(1000) * r)
            r_term_denominator = np.log(1) - (np.log(1000) * 1)

            r_term_numerator = la + np.log1p(-np.exp(r_term_numerator - la))
            r_term_denominator = la + np.log1p(-np.exp(r_term_denominator - la))

            res = a_term + (r_term_numerator - r_term_denominator)
        return res

    def overall_prob_log(self, r, z_term):
        res = np.log(1) - (z_term + (np.log(1000) * (r - 1)))
        return res

    def test_zero_roots(self):

        r = 0
        b_term = 0

        expected = self.naive_way(r, b_term)
        actual = self.closed_form(r, b_term)

        print("Expected: {}, Actual: {}".format(expected, actual))

        np.testing.assert_allclose(expected, actual)

    def test_single_root(self):

        r = 1
        b_term = 1

        expected = self.naive_way(r, b_term)
        actual = self.closed_form(r, b_term)

        print("Expected: {}, Actual: {}".format(expected, actual))

        np.testing.assert_allclose(expected, actual)

        overall = self.overall_prob(r, actual)

        print("overall: {}".format(overall))

    def test_two_roots(self):

        r = 2
        b_term = 2

        expected = self.naive_way(r, b_term)
        actual = self.closed_form(r, b_term)

        print("Expected: {}, Actual: {}".format(expected, actual))

        np.testing.assert_allclose(expected, actual)

        overall = self.overall_prob(r, actual)

        print("overall: {}".format(overall))

    def test_three_roots(self):

        r = 3
        b_term = 3

        expected = self.naive_way(r, b_term)
        actual = self.closed_form(r, b_term)

        print("Expected: {}, Actual: {}".format(expected, actual))

        np.testing.assert_allclose(expected, actual)

        overall = self.overall_prob(r, actual)

        print("overall: {}".format(overall))

    def test_one_vs_two_roots(self):

        b_term = 2

        one_root_z = self.closed_form(1, b_term)
        two_root_z = self.closed_form(2, b_term)

        one_root_overall = self.overall_prob(1, one_root_z)
        two_root_overall = self.overall_prob(2, two_root_z)

        print("One root: {}; Two roots: {}".format(one_root_overall, two_root_overall))

        np.testing.assert_allclose(one_root_overall / 1000, two_root_overall, atol=1e-05)

    def test_log_vs_not(self):

        b_term = 2

        two_root_z = self.closed_form(2, b_term)

        expected_log = np.log(two_root_z)

        two_root_z_log = self.closed_form_log(2, b_term)

        np.testing.assert_allclose(two_root_z_log, expected_log)

        two_root_overall = self.overall_prob(2, two_root_z)
        expected_log_overall = np.log(two_root_overall)

        two_root_overall_log = self.overall_prob_log(2, two_root_z_log)

        np.testing.assert_allclose(two_root_overall_log, expected_log_overall)

    def test_seven_roots(self):

        r = 7
        b_term = 7

        expected = self.naive_way(r, b_term)
        actual = self.closed_form(r, b_term)

        print("Expected: {}, Actual: {}".format(expected, actual))

        np.testing.assert_allclose(expected, actual)

        overall = self.overall_prob(r, actual)

        print("overall: {}".format(overall))

    def test_zero_roots_log(self):

        r = 0
        b_term = 0

        expected = self.closed_form(r, b_term)
        expected_log_z = np.log(expected)
        actual_log_z = self.closed_form_log(r, b_term)

        print("Z-terms || Expected: {}, Actual: {}".format(expected_log_z, actual_log_z))

        np.testing.assert_allclose(actual_log_z, expected_log_z)

        expected_prob = self.overall_prob(r, expected)
        expected_prob_log = np.log(expected_prob)

        actual_prob_log = self.overall_prob_log(r, actual_log_z)

        print("Prob || Expected: {}, Actual: {}".format(expected_prob_log, actual_prob_log))

        np.testing.assert_allclose(actual_prob_log, expected_prob_log)


if __name__ == "__main__":
    unittest.main()
