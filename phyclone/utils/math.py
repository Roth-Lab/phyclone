"""
Created on 8 Dec 2016

@author: Andrew Roth
"""

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import fftconvolve


def bernoulli_rvs(rng: np.random.Generator, p=0.5):
    return rng.random() < p


def discrete_rvs(p, rng):
    p = p / np.sum(p)
    return rng.multinomial(1, p).argmax()


@numba.jit(nopython=True)
def simple_log_factorial(n, arr):
    idxs = np.nonzero(arr == -math.inf)[0]

    for i in idxs:
        if i > n:
            break
        if i == 0:
            arr[i] = np.log(1)
        else:
            arr[i] = np.log(i) + arr[i - 1]


@numba.jit(nopython=True)
def exp_normalize(log_p):
    """Normalize a vector numerically safely.

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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def log_sum_exp(log_X):
    """Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    This implementation is numerically safer than the naive method.
    """
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit(nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


@numba.vectorize()
def log_gamma(x):
    return math.lgamma(x)


@numba.jit(nopython=True)
def log_beta(a, b):
    if a <= 0 or b <= 0:
        return -np.inf

    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.jit(nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@lru_cache(maxsize=None)
def cached_log_factorial(x):
    return log_factorial(x)


@numba.jit(nopython=True)
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


def log_multinomial_coefficient(x):
    """Compute the multinomial coefficient.

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


@numba.jit(nopython=True)
def log_beta_binomial_likelihood(n, x, a, b):
    return log_beta(a + x, b + n - x) - log_beta(a, b)


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def log_binomial_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + log_binomial_likelihood(n, x, p)


@numba.jit(nopython=True)
def log_beta_binomial_pdf(n, x, a, b):
    return log_binomial_coefficient(n, x) + log_beta_binomial_likelihood(n, x, a, b)


@numba.jit(nopython=True, fastmath=True)
def conv_log(log_x, log_y, ans):
    """Direct convolution in log space."""
    n = len(log_x)

    log_y = log_y[::-1]

    for k in range(1, n + 1):
        v_arr = np.empty(k)
        max_val = -np.inf
        for j in range(k):
            curr = log_x[j] + log_y[n - (k - j)]
            v_arr[j] = curr
            if curr > max_val:
                max_val = curr

        v_arr -= max_val

        np.exp(v_arr, v_arr)

        sub_ans = 0
        for i in range(k):
            sub_ans += v_arr[i]

        ans[k - 1] = np.log(sub_ans) + max_val

    return ans


def fft_convolve_two_children(child_1, child_2):
    """FFT convolution"""
    child_1_maxes = np.max(child_1, axis=-1, keepdims=True)

    child_2_maxes = np.max(child_2, axis=-1, keepdims=True)

    child_1_norm = np.exp(child_1 - child_1_maxes)

    child_2_norm = np.exp(child_2 - child_2_maxes)

    result = fftconvolve(child_1_norm, child_2_norm, axes=[-1])

    result = result[..., : child_1_norm.shape[-1]]

    result[result <= 0] = 1e-100

    result = np.log(result, order="C", dtype=np.float64)

    result += child_2_maxes

    result += child_1_maxes

    return result


def non_log_conv(child_log_R, prev_log_D_n):
    """Compute the recursion over D using the numpy."""
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = np.convolve(R_norm, D_norm)

    result = result[: len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result) + log_D_max + log_R_max
