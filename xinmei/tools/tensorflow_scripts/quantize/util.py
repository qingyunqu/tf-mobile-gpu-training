import gzip
import io
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from google.protobuf.message import DecodeError
from google.protobuf.text_format import MessageToString, Parse, ParseError
from tensorflow import GraphDef, NodeDef
from tensorflow.python.framework.tensor_shape import as_shape

__author__ = 'afpro'
__email__ = 'admin@afpro.net'

__all__ = [
    'isclose',
    'reshaped_view',
    'all_same_value',
    'rmse_of',
    'index_of',
    'obfuscate_name',
    'read_message',
    'write_message',
    'prepare_output_dir',
    'replace_graph_node_names',
    'find_node_by_name',
    'copy_node',
    'const_node',
    'str_is_empty',
    'valid_input_name',
]

_alphabet_lower = 'abcdefghijklmnopqrstuvwxyz'
_alphabet_upper = _alphabet_lower.upper()
_name_char_pool = _alphabet_lower + _alphabet_upper

_input_regex = re.compile(r'\^?(?P<NAME>[^:]+)(:.*)?')
_node_name_regex_tpl = r'(s: "loc:@|input: "\^?|name: "\^?)(?P<name>{})[:"]'


def isclose(a, b, rel_tol=1e-9):
    return abs(a - b) <= rel_tol


def reshaped_view(value, shape=(-1,)):
    if len(value.shape) == 0:
        return value

    view = value.view()
    view.shape = shape
    return view


def all_same_value(value):
    if value.size == 0:
        return False, 0
    else:
        view = reshaped_view(value)
        v = view[0]
        for another in view[1:]:
            if not isclose(v, another):
                return False, 0
        return True, v


def _to_chunk(v1, v2, chunk_size):
    return [(v1[s * chunk_size:(s + 1) * chunk_size], v2[s * chunk_size:(s + 1) * chunk_size])
            for s in range(int(math.ceil(v1.size / chunk_size)))]


def rmse_of(v1, v2, concurrent_chunk=0):
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert v1.size == v2.size
    v1 = reshaped_view(v1)
    v2 = reshaped_view(v2)
    if 0 < concurrent_chunk < v1.size:
        e = ThreadPoolExecutor()
        a = e.map(
            lambda v: np.sum(np.square(v[0] - v[1])),
            _to_chunk(v1, v2, concurrent_chunk))
        e.shutdown(wait=True)
        return math.sqrt(sum(list(a)) / v1.size)
    else:
        return math.sqrt(np.mean(np.square(v1 - v2)))


def index_of(value, table):
    d = 0
    idx = None
    for i in range(len(table)):
        if idx is None or abs(table[i] - value) < d:
            idx = i
            d = abs(table[i] - value)
    return idx


def obfuscate_name(index):
    assert index >= 0
    index = index + 1
    with io.StringIO() as s:
        while index > 0:
            s.write(_name_char_pool[index % len(_name_char_pool)])
            index = index // len(_name_char_pool)
        return s.getvalue()


def read_message(path, message):
    def p_text():
        with open(path, 'r') as fp:
            Parse(fp.read(), message)

    def p_binary():
        with gzip.open(path, 'rb') if path.endswith('.pbgz') else open(path, 'rb') as fp:
            message.ParseFromString(fp.read())

    if path.endswith('.pbtxt') or path.endswith('.txt'):
        p = [p_text, p_binary]
    else:
        p = [p_binary, p_text]

    try:
        p[0]()
    except ParseError as pe:
        try:
            p[1]()
        except DecodeError as de:
            raise Exception('parse failed {} {}'.format(pe, de))


def write_message(output, name, message,
                  write_text_message=True,
                  write_binary_message=True,
                  write_compressed_message=True):
    if write_binary_message:
        with open(os.path.join(output, name + '.pb'), 'wb') as fp:
            fp.write(message.SerializeToString())
    if write_text_message:
        with open(os.path.join(output, name + '.pbtxt'), 'w') as fp:
            fp.write(MessageToString(message))
    if write_compressed_message:
        with gzip.open(os.path.join(output, name + '.pbgz'), 'wb') as fp:
            fp.write(message.SerializeToString())


def prepare_output_dir(output='output'):
    if output is None:
        output = os.path.curdir
    elif os.path.isfile(output):
        raise Exception('{} is file, not dir'.format(output))
    if not os.path.exists(output):
        os.makedirs(output)
    return output


def replace_graph_node_names(graph, mapping):
    assert isinstance(mapping, dict)

    # get all nodes, sort by node name length
    all_nodes = [node.name for node in graph.node if len(node.name) > 0]
    all_nodes.sort(key=lambda k: len(k), reverse=True)

    # regex, match all node name
    all_nodes_regex = re.compile(_node_name_regex_tpl.format('|'.join(all_nodes)))

    # old graph text
    graph_text = MessageToString(graph)

    # replace all node name
    obfuscated_graph_text = io.StringIO()
    last_match_end = 0
    while True:
        match = all_nodes_regex.search(graph_text, last_match_end)
        if match is None:
            break

        # prefix
        match_beg, match_end = match.span('name')
        obfuscated_graph_text.write(graph_text[last_match_end:match_beg])
        last_match_end = match_end

        # node name
        node_name = graph_text[match_beg:match_end]
        obfuscated_graph_text.write(mapping.get(node_name, node_name))

    obfuscated_graph_text.write(graph_text[last_match_end:])

    obfuscated_graph = GraphDef()
    Parse(obfuscated_graph_text.getvalue(), obfuscated_graph)
    obfuscated_graph_text.close()
    return obfuscated_graph


def find_node_by_name(graph, name):
    for node in graph.node:
        if node.name == name:
            return node
    raise Exception('node {} not found'.format(name))


def copy_node(node_name, src_graph, node_list):
    if node_name in map(lambda _: _.name, node_list):
        return

    node = find_node_by_name(src_graph, node_name)
    node_list.append(node)
    for input_name in node.input:
        copy_node(valid_input_name(input_name), src_graph, node_list)


def _extend(field, data):
    if len(data.shape) == 0:
        if data.dtype == np.int32:
            field.append(int(data))
        elif data.dtype == np.float32:
            field.append(float(data))
        else:
            raise Exception('unknown dtype {}'.format(data.dtype))
    else:
        field.extend(reshaped_view(data))


def const_node(dtype, value, shape):
    node = NodeDef()
    node.op = 'Const'
    node.attr['dtype'].type = dtype.as_datatype_enum
    node.attr['value'].tensor.dtype = dtype.as_datatype_enum
    node.attr['value'].tensor.tensor_shape.CopyFrom(as_shape(shape).as_proto())

    if value.dtype == np.float32:
        _extend(node.attr['value'].tensor.float_val, value)
    elif value.dtype == np.int32:
        _extend(node.attr['value'].tensor.int_val, value)
    else:
        raise Exception('const_node, unknown dtype {}'.format(value.dtype))
    return node


def str_is_empty(s):
    return s is None or len(s) == 0


def valid_input_name(name):
    return _input_regex.match(name).group('NAME')
