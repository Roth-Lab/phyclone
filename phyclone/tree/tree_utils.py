import numpy as np
import numba
from phyclone.utils import two_np_arr_cache, list_of_np_cache
from phyclone.utils.math import lse_accumulate


@list_of_np_cache(maxsize=4096)
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
    log_S = _sub_compute_S(log_D)

    return np.ascontiguousarray(log_S)


# def _sub_compute_S(log_D):
#     log_S = np.zeros(log_D.shape, order='C')
#     num_dims = log_D.shape[0]
#     for i in range(num_dims):
#         log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])
#     return log_S

@numba.jit(cache=True, nopython=True)
def _sub_compute_S(log_D):
    # log_S = np.zeros(log_D.shape, order='C')
    log_S = np.empty_like(log_D)
    num_dims = log_D.shape[0]
    for i in range(num_dims):
        lse_accumulate(log_D[i, :], log_S[i, :])
    return log_S


def compute_log_D(child_log_R_values):
    if len(child_log_R_values) == 0:
        return 0

    log_D = _comp_log_d_split(child_log_R_values)

    return log_D


def _comp_log_d_split(child_log_R_values):
    num_children = len(child_log_R_values)
    if num_children == 1:
        return child_log_R_values[0]

    log_D = _comp_log_d_internals(child_log_R_values, num_children)
    return log_D


# @numba.jit(cache=True, nopython=True)
def _comp_log_d_internals(child_log_R_values, num_children):
    conv_res = _convolve_two_children(child_log_R_values[0], child_log_R_values[1])
    for j in range(2, num_children):
        conv_res = _convolve_two_children(child_log_R_values[j], conv_res)
    return conv_res


@two_np_arr_cache(maxsize=4096)
def _convolve_two_children(child_1, child_2):
    num_dims = child_1.shape[0]
    res_arr = np.empty_like(child_1)
    _conv_two_children_jit(child_1, child_2, num_dims, res_arr)
    return res_arr


@numba.jit(cache=True, nopython=True, parallel=True)
def _conv_two_children_jit(child_1, child_2, num_dims, res_arr):
    for i in numba.prange(num_dims):
        conv_log(child_1[i, :], child_2[i, :], res_arr[i, :])


@numba.jit(cache=True, nopython=True)
def conv_log(log_x, log_y, ans):
    """ Convolve in log space.
    """
    nx = len(log_x)

    log_y = log_y[::-1]
    n = nx

    for k in range(1, n + 1):
        sub_ans = None
        for j in range(k):
            curr = log_x[j] + log_y[n - (k - j)]
            if sub_ans is None:
                sub_ans = curr
            else:
                # max_val = max(sub_ans, curr)
                # min_val = min(sub_ans, curr)
                if sub_ans > curr:
                    max_val = sub_ans
                    min_val = curr
                else:
                    max_val = curr
                    min_val = sub_ans
                sub_ans = max_val + np.log1p(np.exp(min_val - max_val))

        ans[k - 1] = sub_ans

    return ans


def _cache_ratio(cache_obj):
    try:
        ratio = cache_obj.hits / (cache_obj.hits + cache_obj.misses)
    except ZeroDivisionError:
        ratio = 0.0
    return ratio
