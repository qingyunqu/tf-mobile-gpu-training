import numpy as np
from sklearn.cluster import KMeans

from .util import *

__author__ = 'afpro'
__email__ = 'admin@afpro.net'

__all__ = [
    'average_slice',
    'k_means_slice',
]


def _index_of(value, array):
    d = 0
    idx = None
    for i in range(len(array)):
        if idx is None or abs(array[i] - value) < d:
            idx = i
            d = abs(array[i] - value)
    return idx


def average_slice(value, size, zero_is_special=True, zero_rel_tol=1e-5):
    """
    :type value: np.ndarray
    :type size: int
    :type zero_is_special: bool
    :type zero_rel_tol: float
    :rtype: (float,float,np.ndarray)
    """

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.int32 or value.dtype == np.float32

    index = np.ndarray(shape=value.shape, dtype=np.uint32)
    v_min = np.min(value)
    v_max = np.max(value)

    if zero_is_special:  # zero as special index to zero value
        cluster_size = size - 1
        step = (v_max - v_min) / (cluster_size - 1)
        for i in range(value.size):
            if isclose(value.flat[i], 0, rel_tol=zero_rel_tol):
                index.flat[i] = 0
            else:
                index.flat[i] = round((value.flat[i] - v_min) / step) + 1
        return v_min - step, step, index
    else:
        cluster_size = size
        step = (v_max - v_min) / (cluster_size - 1)
        for i in range(value.size):
            index.flat[i] = round((value.flat[i] - v_min) / step)
        return v_min, step, index


def k_means_slice(value, size, zero_is_special=True, zero_rel_tol=1e-5, n_jobs=1):
    """
    :type value: np.ndarray
    :type size: int
    :type zero_is_special: bool
    :type zero_rel_tol: float
    :type n_jobs: int
    :rtype: (np.ndarray,np.ndarray)
    """
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32 or value.dtype == np.int32

    v_min = np.min(value)
    v_max = np.max(value)

    if zero_is_special and v_min < 0 < v_max:
        has_zero_center = True
    else:
        has_zero_center = False

    k_means = KMeans(n_clusters=size - (1 if has_zero_center else 0), n_jobs=n_jobs)

    if has_zero_center:
        fit_data = []
        for i in range(value.size):
            if not isclose(value.flat[i], 0, rel_tol=zero_rel_tol):
                fit_data.append([i, value.flat[i]])
        fit_data = np.array(fit_data, dtype=np.float32)
        k_means.fit(reshaped_view(fit_data[:, 1], (-1, 1)))

        index = np.ndarray(shape=value.shape, dtype=np.uint32)
        index.fill(0)
        for i in range(len(fit_data)):
            real_index = int(fit_data[i][0])
            if not isclose(value.flat[real_index], 0, rel_tol=zero_rel_tol):
                index.flat[real_index] = k_means.labels_[i]
    else:
        k_means.fit(reshaped_view(value, (-1, 1)))
        index = reshaped_view(k_means.labels_)
    table = reshaped_view(k_means.cluster_centers_)
    if value.dtype == np.int32:
        table = table.round().astype(np.int32)
    return table.astype(value.dtype), index.astype(np.uint32)
