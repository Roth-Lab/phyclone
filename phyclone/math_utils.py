'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
import math
import numba
import numpy as np
import random


def bernoulli_rvs(p=0.5):
    return (random.random() < p)


def discrete_rvs(p):
    p = p / np.sum(p)

    return np.random.multinomial(1, p).argmax()


# TODO: consider turning simple factorial fxns into loop versions since python doesn't like recursion
def simple_factorial(n, arr):
    if n <= 1:
        return 1

    if arr[n] > -math.inf:
        return arr[n]

    arr[n] = n * simple_factorial(n - 1, arr)
    return arr[n]


def simple_log_factorial(n, arr):
    if n <= 1:
        if arr[1] == -math.inf:
            arr[1] = np.log(1)
            arr[0] = np.log(1)
        return 0  # because log(1) = 0

    if arr[n] > -math.inf:
        return arr[n]

    arr[n] = np.log(n) + simple_log_factorial(n - 1, arr)
    return arr[n]

# @numba.jit(cache=True, nopython=True)
# def simple_log_factorial_looped(n, arr):
#     idxs = np.nonzero(arr == -math.inf)[0]
#
#     for i in idxs:
#         if i > n:
#             break
#         if i == 0:
#             arr[i] = np.log(1)
#         else:
#             arr[i] = np.log(i) + arr[i - 1]


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


@numba.jit(nopython=True)
def log_binomial_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + log_binomial_likelihood(n, x, p)


@numba.jit(nopython=True)
def log_beta_binomial_pdf(n, x, a, b):
    return log_binomial_coefficient(n, x) + log_beta_binomial_likelihood(n, x, a, b)
