'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
import math
import numba
import numpy as np
from functools import lru_cache


def bernoulli_rvs(rng: np.random.Generator, p=0.5):
    return rng.random() < p


def discrete_rvs(p, rng):
    p = p / np.sum(p)
    return rng.multinomial(1, p).argmax()


@numba.jit(cache=True, nopython=True)
def simple_log_factorial(n, arr):
    idxs = np.nonzero(arr == -math.inf)[0]

    for i in idxs:
        if i > n:
            break
        if i == 0:
            arr[i] = np.log(1)
        else:
            arr[i] = np.log(i) + arr[i - 1]


@numba.jit(cache=True, nopython=True)
def exp_normalize(log_p):
    """ Normalize a vector numerically safely.

    Parameters
    ----------
    log_p: array_like (float)
        Unnormalized array of values in log space.

    Returns:
    -------
    p: array_like (float)
        Normalized array of values.
    log_norm: float
        Log normalization constant.
    """
    log_norm = log_sum_exp(log_p)

    p = np.exp(log_p - log_norm)

    p = p / p.sum()

    return p, log_norm


@numba.jit(cache=True, nopython=True)
def lse(log_x):
    inf_check = np.all(np.isinf(log_x))
    if inf_check:
        return log_x[0]

    x = log_x[np.isfinite(log_x)]
    ans = x[0]

    for i in range(1, len(x)):
        curr = x[i]
        if ans > curr:
            max_value = ans
            min_value = curr
        else:
            max_value = curr
            min_value = ans
        ans = max_value + np.log1p(np.exp(min_value - max_value))

    return ans


@numba.jit(cache=True, nopython=True)
def lse_accumulate(log_x, out_arr):
    len_arr = len(log_x)
    t = log_x[0]
    out_arr[0] = t
    for i in range(1, len_arr):
        curr = log_x[i]
        if t > curr:
            max_value = t
            min_value = curr
        else:
            max_value = curr
            min_value = t
        t = max_value + np.log1p(np.exp(min_value - max_value))
        out_arr[i] = t
    return out_arr


@numba.jit(cache=True, nopython=True)
def log_sum_exp(log_X):
    """ Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    This implementation is numerically safer than the naive method.
    """
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit(cache=True, nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


@numba.vectorize(["float64(float64)", "int64(float64)"])
def log_gamma(x):
    return math.lgamma(x)


@numba.jit(cache=True, nopython=True)
def log_beta(a, b):
    if a <= 0 or b <= 0:
        return -np.inf

    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.jit(cache=True, nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@lru_cache(maxsize=None)
def cached_log_factorial(x):
    return log_factorial(x)


@numba.jit(cache=True, nopython=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


def log_multinomial_coefficient(x):
    """ Compute the multinomial coefficient.

    Parameters
    ----------
    x: list
        The number of elements in each category.
    """
    if len(x) == 0:
        return 0

    n = sum(x)

    result = log_factorial(n)

    for x_i in x:
        result -= log_factorial(x_i)

    return result


@numba.jit(cache=True, nopython=True)
def log_beta_binomial_likelihood(n, x, a, b):
    return log_beta(a + x, b + n - x) - log_beta(a, b)


@numba.jit(cache=True, nopython=True)
def log_binomial_likelihood(n, x, p):
    if p == 0:
        if x == 0:
            return 0
        else:
            return -np.inf

    if p == 1:
        if x == n:
            return 0
        else:
            return -np.inf

    return x * np.log(p) + (n - x) * np.log(1 - p)


@numba.jit(cache=True, nopython=True)
def log_binomial_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + log_binomial_likelihood(n, x, p)


@numba.jit(cache=True, nopython=True)
def log_beta_binomial_pdf(n, x, a, b):
    return log_binomial_coefficient(n, x) + log_beta_binomial_likelihood(n, x, a, b)


@numba.jit(cache=True, nopython=True)
def conv_log(log_x, log_y, ans):
    """ Convolve in log space.
    """
    nx = len(log_x)

    log_y = log_y[::-1]
    n = nx

    for k in range(1, n + 1):
        # sub_ans = None
        max_val = None
        v_arr = np.empty(k)
        for j in range(k):
            curr = log_x[j] + log_y[n - (k - j)]
            v_arr[j] = curr
            if max_val is None:
                max_val = curr
            else:
                if curr > max_val:
                    max_val = curr
                #     min_val = curr
                # else:
                #     max_val = curr
                #     min_val = sub_ans
                # sub_ans = max_val + np.log1p(np.exp(min_val - max_val))

        # sub_ans = 0
        # for val in v_arr:
        #     sub_ans += np.exp(val - max_val)

        v_arr -= max_val

        v_arr = np.exp(v_arr, v_arr)

        sub_ans = v_arr.sum()

        ans[k - 1] = np.log(sub_ans) + max_val

    return ans
# def conv_log(log_x, log_y, ans):
#     """ Convolve in log space.
#     """
#     nx = len(log_x)
#
#     log_y = log_y[::-1]
#     n = nx
#
#     for k in range(1, n + 1):
#         # sub_ans = None
#         max_val = None
#         v_arr = np.empty(k)
#         for j in range(k):
#             curr = log_x[j] + log_y[n - (k - j)]
#             v_arr[j] = curr
#             if max_val is None:
#                 max_val = curr
#             else:
#                 if curr > max_val:
#                     max_val = curr
#                 #     min_val = curr
#                 # else:
#                 #     max_val = curr
#                 #     min_val = sub_ans
#                 # sub_ans = max_val + np.log1p(np.exp(min_val - max_val))
#
#         sub_ans = 0
#         for val in v_arr:
#             sub_ans += np.exp(val - max_val)
#
#         ans[k - 1] = np.log(sub_ans) + max_val
#
#     return ans
# def conv_log(log_x, log_y, ans):
#     """ Convolve in log space.
#     """
#     nx = len(log_x)
#
#     log_y = log_y[::-1]
#     n = nx
#
#     for k in range(1, n + 1):
#         sub_ans = None
#         for j in range(k):
#             curr = log_x[j] + log_y[n - (k - j)]
#             if sub_ans is None:
#                 sub_ans = curr
#             else:
#                 if sub_ans > curr:
#                     max_val = sub_ans
#                     min_val = curr
#                 else:
#                     max_val = curr
#                     min_val = sub_ans
#                 sub_ans = max_val + np.log1p(np.exp(min_val - max_val))
#
#         ans[k - 1] = sub_ans
#
#     return ans
