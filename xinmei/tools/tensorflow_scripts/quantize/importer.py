import numpy as np
import tensorflow as tf

from .quantized_pb2 import *
from .util import *

__author__ = 'afpro'
__email__ = 'admin@afpro.net'

__all__ = [
    'restore_average_slice',
    'restore_k_means_slice',
    'quantized_raw_item_to_ndarray',
    'quantized_simple_item_to_ndarray',
    'quantized_k_means_item_to_ndarray',
    'quantized_item_to_ndarray',
    'load_graph',
    'feeds_of_graph',
    'import_graph',
]


def restore_average_slice(base, step, index, dtype=np.float32, zero_is_special=True):
    v = np.ndarray(shape=index.shape, dtype=dtype)
    for i in range(v.size):
        v.flat[i] = 0 if zero_is_special and isclose(0, index.flat[i]) else (index.flat[i] * step + base)
    return v


def restore_k_means_slice(table, index):
    return np.take(table, index)


def quantized_raw_item_to_ndarray(item):
    if item.dtype == tf.int32.as_datatype_enum:
        return reshaped_view(np.array(item.int_raw), item.shape) + int(item.base)
    elif item.dtype == tf.float32.as_datatype_enum:
        return reshaped_view(np.array(item.float_raw), item.shape) + float(item.base)
    else:
        raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_simple_item_to_ndarray(item):
    if item.dtype == tf.int32.as_datatype_enum:
        return reshaped_view(restore_average_slice(item.base, item.step, np.array(item.index), dtype=np.int32),
                             item.shape)
    elif item.dtype == tf.float32.as_datatype_enum:
        return reshaped_view(restore_average_slice(item.base, item.step, np.array(item.index), dtype=np.float32),
                             item.shape)
    else:
        raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_k_means_item_to_ndarray(item):
    if item.dtype == tf.int32.as_datatype_enum:
        return reshaped_view(restore_k_means_slice(item.int_table, item.index), item.shape)
    elif item.dtype == tf.float32.as_datatype_enum:
        return reshaped_view(restore_k_means_slice(item.float_table, item.index), item.shape)
    else:
        raise Exception('unknown dtype {}'.format(item.dtype))


def quantized_item_to_ndarray(item):
    if item.vtype == RAW:
        return quantized_raw_item_to_ndarray(item)
    elif item.vtype == SIMPLE:
        return quantized_simple_item_to_ndarray(item)
    elif item.vtype == TABLE:
        return quantized_k_means_item_to_ndarray(item)
    else:
        raise Exception('unknown vtype {}'.format(item.vtype))


def load_graph(path):
    g = QuantizedGraph()
    read_message(path, g)
    return g


def feeds_of_graph(graph):
    feeds = dict()
    for item in graph.items:
        feeds['{}:0'.format(item.name)] = quantized_item_to_ndarray(item)
    return feeds


def import_graph(path, name=None):
    graph = load_graph(path)
    tf.import_graph_def(graph.graph, name=name)
    return feeds_of_graph(graph)
