from phyclone.smc.kernels.fully_adapted import _get_cached_full_proposal_dist
from phyclone.smc.kernels.semi_adapted import _get_cached_semi_proposal_dist
from phyclone.tree.utils import compute_log_S, _convolve_two_children


# Utils for dev debugging and performance checking

def _cache_ratio(cache_obj):
    try:
        ratio = cache_obj.hits / (cache_obj.hits + cache_obj.misses)
    except ZeroDivisionError:
        ratio = 0.0
    return ratio


def create_cache_info_file(out_file):
    with open(out_file, "w") as f:
        print(
            "compute_s cache info: {}, hit ratio: {}".format(
                compute_log_S.cache_info(), _cache_ratio(compute_log_S.cache_info())
            ),
            file=f,
        )
        print(
            "_convolve_two_children cache info: {}, hit ratio: {}".format(
                _convolve_two_children.cache_info(),
                _cache_ratio(_convolve_two_children.cache_info()),
            ),
            file=f,
        )
        print(
            "_get_cached_proposal_dist cache info: {}, hit ratio: {}".format(
                _get_cached_full_proposal_dist.cache_info(),
                _cache_ratio(_get_cached_full_proposal_dist.cache_info()),
            ),
            file=f,
        )
        print(
            "_get_cached_semi_proposal_dist cache info: {}, hit ratio: {}".format(
                _get_cached_semi_proposal_dist.cache_info(),
                _cache_ratio(_get_cached_semi_proposal_dist.cache_info()),
            ),
            file=f,
        )
