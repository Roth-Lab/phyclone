import numba
import numpy as np
import time
from functools import lru_cache, wraps
# import hashlib
import xxhash


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)

        set_numba_seed(seed)


@numba.jit(nopython=True)
def set_numba_seed(seed):
    np.random.seed(seed)


class Timer:
    """ Taken from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html
    """

    def __init__(self, func=time.time):
        self.elapsed = 0.0

        self._func = func

        self._start = None

    @property
    def running(self):
        return self._start is not None

    def reset(self):
        self.elapsed = 0.0

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')

        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')

        end = self._func()

        self.elapsed += end - self._start

        self._start = None

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()


def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

    """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            # hashable_array = array_to_tuple(np_array)
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # def array_to_tuple(np_array):
        #     """Iterates recursivelly."""
        #     try:
        #         return tuple(array_to_tuple(_) for _ in np_array)
        #     except TypeError:
        #         return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


# def list_of_np_cache(*args, **kwargs):
#     """LRU cache implementation for functions whose FIRST parameter is a list of numpy arrays
#     >>> array = np.array([[1, 2, 3], [4, 5, 6]])
#     >>> @list_of_np_cache(maxsize=256)
#     ... def multiply(array, factor):
#     ...     print("Calculating...")
#     ...     return factor*array
#     >>> multiply(array, 2)
#     Calculating...
#     array([[ 2,  4,  6],
#            [ 8, 10, 12]])
#     >>> multiply(array, 2)
#     array([[ 2,  4,  6],
#            [ 8, 10, 12]])
#     >>> multiply.cache_info()
#     CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
#
#     """
#
#     def decorator(function):
#
#         @wraps(function)
#         def wrapper(list_of_np_array, *args, **kwargs):
#             # hashable_array = array_to_tuple(np_array)
#             hashable_set = frozenset(tuple(map(tuple, np_array)) for np_array in list_of_np_array)
#             # hashable_array = tuple(map(tuple, set_of_np_array))
#             return cached_wrapper(hashable_set, *args, **kwargs)
#
#         @lru_cache(*args, **kwargs)
#         def cached_wrapper(hashable_set, *args, **kwargs):
#             array = np.array(list(hashable_set))
#             # array = np.fromiter((np.array(arr_tuple) for arr_tuple in hashable_set),
#             #                     dtype=float, count=len(hashable_set))
#             return function(array, *args, **kwargs)
#
#         # def array_to_tuple(np_array):
#         #     """Iterates recursivelly."""
#         #     try:
#         #         return tuple(array_to_tuple(_) for _ in np_array)
#         #     except TypeError:
#         #         return np_array
#
#         # copy lru_cache attributes over too
#         wrapper.cache_info = cached_wrapper.cache_info
#         wrapper.cache_clear = cached_wrapper.cache_clear
#
#         return wrapper
#
#     return decorator

class YetAnotherWrapper:
    def __init__(self, x) -> None:
        self.values = x
        # self.values = np.array(x, order='C')
        # here you can use your own hashing function
        # tmp = frozenset(hashlib.sha224(np_array.view()).hexdigest() for np_array in x)
        # self.h = hashlib.sha224(x.view()).hexdigest()
        # self.h = (tmp, x.shape)
        self.h = YetAnotherWrapper._create_hashable(x)

    @staticmethod
    def _create_hashable(list_of_np_arrays):
        # hashable = np.array([hashlib.sha224(arr.view()).hexdigest() for arr in list_of_np_arrays], order='C')
        # hashable.sort()
        # ret = hashlib.sha224(hashable).hexdigest()
        hashable = np.array([xxhash.xxh3_64_hexdigest(arr) for arr in list_of_np_arrays], order='C')
        hashable.sort()
        ret = xxhash.xxh3_64_hexdigest(hashable)
        return ret

    def __hash__(self) -> int:
        return hash(self.h)

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h


def list_of_np_cache(*args, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(list_of_np_array, *args, **kwargs):
            # hashable_array = array_to_tuple(np_array)
            # hashable_set = frozenset(tuple(map(tuple, np_array)) for np_array in list_of_np_array)
            # hashable_array = tuple(map(tuple, set_of_np_array))
            wrapped_obj = YetAnotherWrapper(list_of_np_array)
            return cached_wrapper(wrapped_obj, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_set, *args, **kwargs):
            # array = np.array(hashable_set.values)
            # array = np.fromiter((np.array(arr_tuple) for arr_tuple in hashable_set),
            #                     dtype=float, count=len(hashable_set))
            # array = hashable_set.values
            array = np.array(hashable_set.values, order='C')
            return function(array, *args, **kwargs)

        # def array_to_tuple(np_array):
        #     """Iterates recursivelly."""
        #     try:
        #         return tuple(array_to_tuple(_) for _ in np_array)
        #     except TypeError:
        #         return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator
