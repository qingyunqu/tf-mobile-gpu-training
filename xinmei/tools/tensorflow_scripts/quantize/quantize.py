import copy
import math
import os
import re

import click
import numpy as np
import tensorflow as tf
from tensorflow import Graph, GraphDef, NodeDef, Session, as_dtype, import_graph_def
from tensorflow.python.framework.tensor_shape import as_shape
from tensorflow.python.framework.tensor_util import constant_value

from . import quantized_pb2
from .importer import *
from .slice import *
from .util import *
from ..isolator import isolator

__author__ = 'afpro'
__email__ = 'admin@afpro.net'

__all__ = [
    'obfuscate_graph',
    'Quantize',
]

while_node_name_reg = re.compile(r'(^|.*/)while(_\d+)?($|/.*)')


def _is_quantifiable_type(dtype):
    return dtype == np.float32


def obfuscate_graph(graph, mapping=None):
    mapped = {}
    if mapping is not None:
        all_mapping_v = set()
        for mapping_k, mapping_v in mapping.items():
            if mapping_v in all_mapping_v:
                raise RuntimeError(f'mapping has dup value: {mapping_k} => {mapping_v}')
            all_mapping_v.add(mapping_v)
        del all_mapping_v

    out_of_range_id = len(graph.node)
    current_id = 0
    for node in graph.node:
        if len(node.name) == 0:
            continue
        if mapping is not None and node.name in mapping:
            obfuscated = mapping[node.name]
        else:
            obfuscated = obfuscate_name(current_id)
        current_id += 1

        while obfuscated in mapped.values():
            obfuscated = obfuscate_name(out_of_range_id)
            out_of_range_id += 1

        mapped[node.name] = obfuscated

    return replace_graph_node_names(graph, mapped), mapped


def _extend(container, value):
    if value.size > 0:
        container.extend(reshaped_view(value))
    elif value.dtype == np.int32:
        container.extend([int(value)])
    elif value.dtype == np.float32:
        container.extend([float(value)])
    else:
        raise Exception('unknown dtype {}'.format(value.dtype))


class Quantize:
    def __init__(self, origin_graph_path, init_node=None, log_fn=None):
        self._log_fn = log_fn
        self._obfuscate_names = {}
        self._output_dependencies = {}

        self._tf_graph_def = GraphDef()
        read_message(origin_graph_path, self._tf_graph_def)

        self._processed_node_names = set()
        self._processed_nodes = []
        self._quantized_data = {}

        self._value_cache = {}

        self._session = Session(graph=Graph())
        self._init(init_node)

    @isolator
    def session(self):
        return self._session

    @session
    def _init(self, init_node):
        import_graph_def(self._tf_graph_def, name='')
        if init_node is not None:
            self._session.run(init_node)

    def _log(self, msg):
        if self._log_fn is not None:
            self._log_fn(msg)

    def _find_node_by_name(self, name):
        return find_node_by_name(self._tf_graph_def, name)

    def _datatype_enum_of_node(self, node_name, default_value):
        tensor = self._session.graph.get_tensor_by_name('{}:0'.format(node_name))
        if tensor is None:
            return as_dtype(default_value).as_datatype_enum
        else:
            return tensor.dtype.as_datatype_enum

    def _fill_raw(self, item, value):
        self._log('\traw')
        base = np.min(value) if value.dtype == np.float32 else int(np.mean(value))
        data = value - base
        if value.dtype == np.int32:
            _extend(item.int_raw, data)
        elif value.dtype == np.float32:
            _extend(item.float_raw, data)
        else:
            raise Exception('unknown dtype {}'.format(value.dtype.name))
        item.vtype = quantized_pb2.RAW
        item.base = base

    def _fill_simple(self, item, base, step, index):
        self._log('\tsimple')
        item.vtype = quantized_pb2.SIMPLE
        item.base = base
        item.step = step
        _extend(item.index, index)

    def _fill_table(self, item, table, index):
        self._log('\ttable')
        item.vtype = quantized_pb2.TABLE
        if table.dtype == np.int32:
            _extend(item.int_table, table)
        elif table.dtype == np.float32:
            _extend(item.float_table, table)
        else:
            raise Exception('unknown dtype {}'.format(table.dtype))
        _extend(item.index, index)

    def _fill(self, item, value, raw_only):
        if raw_only:
            self._fill_raw(item, value)
            return

        fallback_base, fallback_step, fallback_index = average_slice(value, 256)
        fallback_rmse = rmse_of(value,
                                restore_average_slice(fallback_base, fallback_step, fallback_index, dtype=value.dtype))
        rmse_threshold = min(np.std(value) * 0.05, fallback_rmse)
        self._log('\t256 average slice rmse {}'.format(fallback_rmse))
        self._log('\trmse threshold {}'.format(rmse_threshold))
        bit_size = 2
        while bit_size < 9:
            k_mean_table, k_means_index = k_means_slice(value, 2 ** bit_size, n_jobs=-1)
            k_means_rmse = rmse_of(value, restore_k_means_slice(k_mean_table, k_means_index),
                                   concurrent_chunk=10000)
            self._log('\tk-means table {}, rmse {}'.format(2 ** bit_size, k_means_rmse))
            if k_means_rmse <= rmse_threshold:
                self._fill_table(item, k_mean_table, k_means_index)
                return
            diff = int(math.floor(math.log2(k_means_rmse / rmse_threshold)))
            if diff > 8 - bit_size:
                diff = 8 - bit_size
            if diff < 1:
                diff = 1
            bit_size += diff

        if fallback_rmse <= rmse_threshold:
            self._fill_simple(item, fallback_base, fallback_step, fallback_index)
            return

        self._fill_raw(item, value)

    def _quantize_value(self, quantize, skip_quantize, name, value):
        self._log('quantize {}: {} {}'.format(name, value.dtype, value.shape))
        item = quantized_pb2.QuantizedItem()
        item.name = name
        item.dtype = self._datatype_enum_of_node(name, value.dtype)
        item.shape.extend(value.shape)
        self._fill(item, value, not quantize or (skip_quantize is not None and name in skip_quantize))
        self._log('\tdone')
        return item

    def _keep_node(self, node_name, deps):
        node = self._find_node_by_name(node_name)
        if node.op == 'Placeholder' or node.op == 'PlaceholderWithDefault':
            deps.add(node_name)

        # inputs
        if node_name in self._processed_node_names:
            return
        self._processed_node_names.add(node_name)

        for input_node in node.input:
            self._keep_node(valid_input_name(input_node), deps)

        # node
        processed_node = None
        node_tensor = None
        node_value = None

        # process
        if node.op in {'Const', 'VariableV2'}:
            with self._session.graph.as_default(), self._session.as_default():
                try:
                    node_tensor = self._session.graph.get_tensor_by_name('{}:0'.format(node_name))
                    node_value = constant_value(node_tensor) if node.op == 'Const' else self._session.run(node_tensor)
                except RuntimeError:
                    pass
                except TypeError:
                    pass
                except ValueError:
                    pass

        if node_value is not None and _is_quantifiable_type(node_value.dtype):
            self._log('const op={} {}'.format(node.op, node_name))
            if while_node_name_reg.match(node.name):
                processed_node = node
            elif node_value.size > 256:
                all_same, same_value = all_same_value(node_value)
                if all_same:
                    self._log('const node {} all same value {}'.format(node_name, same_value))
                    processed_node = const_node(node_tensor.dtype, np.array([same_value]), node_value.shape)
                else:
                    processed_node = NodeDef()
                    processed_node.op = 'Placeholder'
                    processed_node.attr['dtype'].type = node_tensor.dtype.as_datatype_enum
                    processed_node.attr['shape'].shape.CopyFrom(as_shape(node_value.shape).as_proto())
                    self._quantized_data[node_name] = node_value
                pass
            else:
                processed_node = const_node(node_tensor.dtype, node_value, node_value.shape)

        # use origin node, if not processed
        if processed_node is None:
            processed_node = NodeDef()
            processed_node.CopyFrom(node)

        # setup name
        processed_node.name = node_name
        self._processed_nodes.append(processed_node)

    def keep_output(self, keep_node_name):
        deps = set()
        self._keep_node(keep_node_name, deps)
        self._output_dependencies[keep_node_name] = deps

    def keep_name(self, node_name, obfuscated_name=None):
        self._obfuscate_names[node_name] = node_name if str_is_empty(obfuscated_name) else obfuscated_name

    def run(self, obfuscate=True, quantize=True, skip_quantize=None):
        graph = quantized_pb2.QuantizedGraph()
        graph.graph.versions.CopyFrom(self._tf_graph_def.versions)
        graph.graph.node.extend(self._processed_nodes)

        self._log('start quantize, data set = {}'.format(len(self._quantized_data)))
        graph.items.extend(
            map(lambda quantized_data: self._quantize_value(quantize, skip_quantize, *quantized_data),
                self._quantized_data.items()))
        self._log('quantized')

        mapping = None
        if obfuscate:
            self._log('start obfuscate')
            obfuscated_graph, mapping = obfuscate_graph(graph.graph, self._obfuscate_names)
            graph.graph.CopyFrom(obfuscated_graph)
            for item in graph.items:
                if item.name in mapping:
                    item.name = mapping[item.name]
            self._log('obfuscated')

        for dep in self._output_dependencies.items():
            out_name = dep[0]
            out_dep_nodes = dep[1]
            if mapping is not None and out_name in mapping:
                self._log('output node: {} => {}'.format(out_name, mapping[out_name]))
            else:
                self._log('output node: {}'.format(out_name))

            for out_dep_node in out_dep_nodes:
                if mapping is not None and out_dep_node in mapping and out_dep_node != mapping[out_dep_node]:
                    self._log('\tdep {} => {}'.format(out_dep_node, mapping[out_dep_node]))
                else:
                    self._log('\tdep {}'.format(out_dep_node))

        self._log('finish')
        return copy.deepcopy(self._output_dependencies), mapping, graph


@click.command(name='quantize')
@click.argument('origin_graph_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--output', type=click.Path(file_okay=False))
@click.option('--obfuscate/--no-obfuscate', default=True)
@click.option('--quantize/--no-quantize', default=True)
@click.option('--output-node', type=click.STRING, multiple=True)
@click.option('--keep', type=click.STRING, multiple=True)
@click.option('--skip-quantize', type=click.STRING, multiple=True)
@click.option('--init-node', default=None)
def main(origin_graph_path, output, obfuscate, quantize, output_node, keep, skip_quantize, init_node):
    dir(tf.contrib)
    q = Quantize(origin_graph_path, init_node, log_fn=print)
    for output_node in output_node:
        q.keep_output(output_node)
    for one_keep in keep:
        keep_parts = one_keep.split(':')
        if len(keep) == 0 or len(keep_parts) > 2:
            raise Exception('invalid keep {}, should be "NAME[:NEW_NAME]"'.format(keep))
        q.keep_name(keep_parts[0], None if len(keep_parts) == 1 else keep_parts[1])

    dep_dict, mapping, graph = q.run(obfuscate=obfuscate, quantize=quantize, skip_quantize=skip_quantize)
    output = prepare_output_dir(output)
    if mapping is not None and len(mapping) > 0:
        with open(os.path.join(output, 'mapping.txt'), 'w') as fp:
            for map_src, map_dst in mapping.items():
                fp.write('{} {}\n'.format(map_src, map_dst))
        print('mapping.txt wrote')
    write_message(output, 'combined', graph)
    with open(os.path.join(output, 'skeleton.graph.pb'), 'wb') as fp:
        fp.write(graph.graph.SerializeToString())
    print('combined.pb* wrote')


if __name__ == '__main__':
    main()
