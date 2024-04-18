import numpy as np
import numba
from phyclone.utils import two_np_arr_cache, list_of_np_cache
from phyclone.utils.math import conv_log, fft_convolve_two_children, non_log_conv


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


def _sub_compute_S(log_D):
    log_S = np.zeros(log_D.shape, order='C')
    num_dims = log_D.shape[0]
    for i in range(num_dims):
        log_S[i, :] = np.logaddexp.accumulate(log_D[i, :])
    return log_S


def compute_log_D(child_log_R_values):
    num_children = len(child_log_R_values)

    if num_children == 0:
        return 0

    if num_children == 1:
        return child_log_R_values[0]

    conv_res = _convolve_two_children(child_log_R_values[0], child_log_R_values[1])
    for j in range(2, num_children):
        conv_res = _convolve_two_children(child_log_R_values[j], conv_res)

    log_D = conv_res
    return log_D


@two_np_arr_cache(maxsize=4096)
def _convolve_two_children(child_1, child_2):
    grid_size = child_1.shape[-1]
    if grid_size < 1000:
        # num_dims = child_1.shape[0]
        # res_arr = np.empty_like(child_1)
        # _conv_two_children_jit(child_1, np.fliplr(child_2), num_dims, res_arr)
        res_arr = _np_conv_dims(child_1, child_2)
    else:
        res_arr = fft_convolve_two_children(child_1, child_2)
    return res_arr


def _np_conv_dims(child_1, child_2):
    num_dims = child_1.shape[0]
    log_D = child_1.copy()
    for i in range(num_dims):
        log_D[i, :] = non_log_conv(child_2[i, :], log_D[i, :])
    return log_D


@numba.jit(cache=True, nopython=True, parallel=True)
def _conv_two_children_jit(child_1, child_2, num_dims, res_arr):
    for i in numba.prange(num_dims):
        conv_log(child_1[i, :], child_2[i, :], res_arr[i, :])


def get_clades(tree):
    result = set()

    for root in tree.roots:
        _clades(result, root, tree)

    return frozenset(result)


def _clades(clades, node, tree):
    current_clade = set()

    for mutation in tree.get_data(node):
        current_clade.add(mutation.idx)

    for child in tree.get_children(node):
        for mutation in _clades(clades, child, tree):
            current_clade.add(mutation)

    clades.add(frozenset(current_clade))

    return current_clade
