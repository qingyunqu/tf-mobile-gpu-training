import os
import re
import sys

import click
from tensorflow.python.tools.selective_registration_header_lib import get_header_from_ops_and_kernels, \
    get_ops_and_kernels


class IncScanner:
    inc_reg = re.compile(r'^\s*#\s*include\s*["<](tensorflow/core/kernels/)?(?P<name>[^/]+)[>"].*$')
    special_cases = {
        'gather_nd_op_cpu_impl.h': 'gather_nd_op_cpu_impl_',
        'mirror_pad_op_cpu_impl.h': 'mirror_pad_op_cpu_impl_',
        'scatter_nd_op_cpu_impl.h': 'scatter_nd_op_cpu_impl_',
        'slice_op_cpu_impl.h': 'slice_op_cpu_impl_',
        'strided_slice_op_impl.h': 'strided_slice_op_inst_',
        'split_op.cc': 'split_lib_cpu.cc',
        'tile_ops.cc': 'tile_ops_cpu',
    }
    name_suffix = {'h', 'cc'}
    name_reg = re.compile(f'(?P<name>[^.]+)\\.(?P<suffix>[^.]+)')
    black_list = {'ops_util.cc'}

    def __init__(self, tf_dir):
        self._inc = None
        self._tf_dir = tf_dir
        self._tf_kernel_dir = None

    def _add(self, name):
        if self._inc is None:
            self._inc = {name}
        else:
            self._inc.add(name)

    @property
    def _kernel_dir(self):
        if self._tf_kernel_dir is None:
            self._tf_kernel_dir = os.path.join(self._tf_dir, 'tensorflow', 'core', 'kernels')
        return self._tf_kernel_dir

    def _contains(self, name):
        return self._inc is not None and name in self._inc

    @property
    def all(self):
        if self._inc is None:
            return
        for name in self._inc:
            yield name

    def clear(self):
        self._inc = None

    def scan_all_suffix(self, name):
        for suffix in IncScanner.name_suffix:
            self.scan(f'{name}.{suffix}')

    def scan(self, name):
        # check dup
        if self._contains(name) or name in IncScanner.black_list:
            return

        # check exist
        path = os.path.join(self._kernel_dir, name)
        if not os.path.isfile(path):
            return

        # add self
        self._add(name)

        # special case
        if name in IncScanner.special_cases:
            for f in os.listdir(self._kernel_dir):
                if f.startswith(IncScanner.special_cases[name]):
                    self.scan(f)

        # header <-> source
        name_match = IncScanner.name_reg.match(name)
        if name_match:
            for suffix in IncScanner.name_suffix:
                if suffix == name_match.group('suffix'):
                    continue
                self.scan(f'{name_match.group("name")}.{suffix}')
            if name_match.group('suffix') == 'h':
                self.scan_all_suffix(f'{name_match.group("name")}_cpu')
            if name_match.group('name').endswith('_op'):
                self.scan_all_suffix(f'{name_match.group("name")[:-3]}_functor')

        # find include statement
        with open(path) as fp:
            for line in fp:
                match = IncScanner.inc_reg.match(line)
                if not match:
                    continue
                self.scan(match.group('name'))


@click.command()
@click.option('--all_ops', type=click.Path(), default=os.path.join(os.path.dirname(__file__), 'all_ops'))
@click.option('--default_op', type=str, default='NoOp:NoOp,_Recv:RecvOp,_Send:SendOp,_Arg:ArgOp,_Retval:RetvalOp')
@click.option('--graph', type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True), multiple=True)
@click.option('--header_out', type=click.File(mode='w'), default=sys.stdout)
@click.option('--kernels_out', type=click.File(mode='w'), default=sys.stderr)
@click.option('--tf_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True), default=None)
def cli(all_ops, default_op, graph, header_out, kernels_out, tf_dir):
    op_def = {}
    with open(all_ops) as fp:
        for op_line in fp:  # type: str
            op_line_parts = op_line.split('"')
            if len(op_line_parts) != 3:
                continue
            stripped_class_name = op_line_parts[0].replace(' ', '')
            src = op_line_parts[1]
            op_def[stripped_class_name] = src

    ops_and_kernels = get_ops_and_kernels('rawproto', graph, default_op)
    header_content = get_header_from_ops_and_kernels(ops_and_kernels, False)
    header_out.write(header_content)

    src_dict = {}
    inc_dict = {}
    inc_scanner = None if tf_dir is None else IncScanner(tf_dir)

    for op, kernel in ops_and_kernels:
        stripped_class_name = kernel.replace(' ', '')
        src = op_def[stripped_class_name]  # type: str
        last_sep = src.rfind('/')
        src_name = f'//{src[:last_sep]}:{src[last_sep+1:]}'
        if src_name in src_dict:
            op_set, class_name_set = src_dict[src_name]
            op_set.add(op)
            class_name_set.add(stripped_class_name)
        else:
            src_dict[src_name] = ({op}, {stripped_class_name})

        if inc_scanner is not None:
            inc_scanner.clear()
            inc_scanner.scan(os.path.basename(src))
            for inc in inc_scanner.all:
                if inc in inc_dict:
                    inc_dict[inc].add(stripped_class_name)
                else:
                    inc_dict[inc] = {stripped_class_name}

    src_list = list(src_dict.items())
    src_list.sort(key=lambda _: _[0])

    avoid_dup_set = set()

    for src_name, (op_set, class_name_set) in src_list:
        if src_name in avoid_dup_set:
            continue
        avoid_dup_set.add(src_name)
        print(f'"{src_name}", # Op"{",".join(op_set)}", Class"{",".join(class_name_set)}"', file=kernels_out)

    inc_list = list(inc_dict.items())
    inc_list.sort(key=lambda _: _[0])

    for inc, clz in inc_list:
        src_name = f'//tensorflow/core/kernels:{inc}'
        if src_name in avoid_dup_set:
            continue
        avoid_dup_set.add(src_name)
        print(f'"{src_name}", # dep by: {",".join(clz)}', file=kernels_out)


if __name__ == '__main__':
    cli()
