import numpy as np
from scipy import fft
from scipy.special import logsumexp
import pyfftw

from phyclone.utils import two_np_arr_cache, list_of_np_cache


def get_set_hash(datapoints_set):
    ret = frozenset(datapoints_set)
    return ret


@two_np_arr_cache(maxsize=1024)
def add_to_log_p(log_p, data_arr):
    return np.add(log_p, data_arr, order='C')


@two_np_arr_cache(maxsize=1024)
def subtract_from_log_p(log_p, data_arr):
    return np.subtract(log_p, data_arr, order='C')


@two_np_arr_cache(maxsize=1024)
def compute_log_R(log_p, log_s):
    return np.add(log_p, log_s, order='C')


@list_of_np_cache(maxsize=1024)
def compute_log_S(child_log_R_values):
    """ Compute log(S) recursion.

    Parameters
    ----------
    child_log_R_values: ndarray
        log_R values from child nodes.
    """
    if len(child_log_R_values) == 0:
        return 0.0

    log_D = compute_log_D(child_log_R_values)

    log_S = np.zeros(log_D.shape, order='C')

    num_dims = log_D.shape[0]

    for i in range(num_dims):
        log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])

    return log_S


def compute_log_D(child_log_R_values):
    if len(child_log_R_values) == 0:
        return 0

    if child_log_R_values[0].size >= 1000:
        log_D = _comp_log_d_split_pyfft(child_log_R_values)
    else:
        log_D = _comp_log_d_split(child_log_R_values)

    return log_D


def _comp_log_d_split(child_log_R_values):
    num_children = len(child_log_R_values)
    if num_children == 1:
        return child_log_R_values[0].copy()

    log_D = child_log_R_values[0].copy()
    num_dims = log_D.shape[0]
    num_children = child_log_R_values.shape[0]

    for j in range(1, num_children):
        child_log_R = child_log_R_values[j]
        for i in range(num_dims):
            log_D[i, :] = _compute_log_D_n(child_log_R[i, :], log_D[i, :])
    return log_D


def _comp_log_d_split_pyfft(child_log_R_values):
    num_children = len(child_log_R_values)
    if num_children == 1:
        return child_log_R_values[0].copy()

    log_D = child_log_R_values[0]

    for j in range(1, num_children):
        log_D = _comp_log_d_fft(log_D, child_log_R_values[j])

    return log_D


def exp_normalize_nary(log_p):
    log_norm = logsumexp(log_p, axis=-1, keepdims=True)

    p = np.exp(log_p - log_norm)

    p = p / p.sum(axis=-1, keepdims=True)

    return p, log_norm


def _comp_log_d_fft(child_log_r_values_1, child_log_r_values_2, ):
    child_log_r_values_norm_1, maxes_1 = exp_normalize_nary(child_log_r_values_1)

    child_log_r_values_norm_2, maxes_2 = exp_normalize_nary(child_log_r_values_2)

    relevant_axis_length = child_log_r_values_norm_1.shape[-1]

    # delta = 1 / (relevant_axis_length - 1)

    outlen = relevant_axis_length + relevant_axis_length - 1

    pad_to = fft.next_fast_len(outlen, real=True)

    with fft.set_backend(pyfftw.interfaces.scipy_fft):
        fwd = fft.rfft(child_log_r_values_norm_1, n=pad_to, axis=-1)

        fwd_2 = fft.rfft(child_log_r_values_norm_2, n=pad_to, axis=-1)

        c_fft = fwd * fwd_2

        log_d = fft.irfft(c_fft, n=pad_to, axis=-1)

    log_d = log_d[..., :relevant_axis_length]

    log_d[log_d <= 0] = 1e-314

    log_d = np.log(log_d, order='C', dtype=np.float64)

    log_d += maxes_1
    log_d += maxes_2

    return log_d


# def _comp_log_d_fft(child_log_R_values):
#     num_children = len(child_log_R_values)
#
#     if num_children == 1:
#         return child_log_R_values[0].copy()
#
#     maxes = np.max(child_log_R_values, axis=-1, keepdims=True)
#     child_log_R_values_norm = np.expm1(child_log_R_values - maxes)
#
#     relevant_axis_length = child_log_R_values.shape[-1]
#
#     outlen = relevant_axis_length + relevant_axis_length - 1
#
#     pad_to = fft.next_fast_len(outlen, real=True)
#
#     fwd = fft.rfft(child_log_R_values_norm, n=pad_to, axis=-1)
#
#     c_fft = fwd * fwd
#
#     log_D = fft.irfft(c_fft, n=pad_to, axis=-1)
#
#     log_D = log_D[..., :relevant_axis_length]
#
#     log_D = np.log1p(log_D) + maxes
#     log_D = np.add.reduce(log_D)
#
#     return log_D


def _compute_log_D_n(child_log_R, prev_log_D_n):
    """ Compute the recursion over D not using the FFT.
    """
    log_R_max = child_log_R.max()

    log_D_max = prev_log_D_n.max()

    R_norm = np.exp(child_log_R - log_R_max)

    D_norm = np.exp(prev_log_D_n - log_D_max)

    result = np.convolve(R_norm, D_norm)

    result = result[:len(child_log_R)]

    result[result <= 0] = 1e-100

    return np.log(result, order='C', dtype=np.float64) + log_D_max + log_R_max
