import re
import os
import click
from tensorflow import GraphDef

default_src = [
    'sendrecv_ops.h',
    'sendrecv_ops.cc',
]

default_op = 'NoOp|_Recv|RecvOp|_Send|SendOp'


@click.command()
@click.argument('kernel_root', is_eager=True, type=click.Path(exists=True, file_okay=False))
@click.argument('model', is_eager=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--additional_src', multiple=True)
@click.option('--filegroup_name', multiple=False, default='kernels_src')
def main(kernel_root, model, additional_src, filegroup_name):
    g = GraphDef()
    with open(model, 'rb') as fp:
        g.ParseFromString(fp.read())

    ops = set([node.op for node in g.node])
    reg = default_op
    for op in ops:
        reg = reg + '|' + op if len(reg) > 0 else op
    reg = 'Name\\("({0})"\\)'.format(reg)
    reg = re.compile(reg)

    inc = re.compile('#include "(tensorflow/core/kernels/)?(?P<name>[a-zA-Z0-9_]*\\.h)"')

    srcs = [
    ]

    def process(file, top=False):
        if file in srcs:
            return
        path = kernel_root + '/' + file
        if not os.path.isfile(path):
            return
        with open(kernel_root + '/' + file, 'r') as fp:
            text = fp.read()
        if top and not reg.search(text):
            return
        srcs.append(file)
        match = inc.search(text)
        while match:
            process(match.group('name'))
            match = inc.search(text, match.span()[1])
        if file.endswith('.cc'):
            if os.path.isfile(kernel_root + '/' + file[:-2] + 'h'):
                process(file[:-2] + 'h')
        if file.endswith('.h'):
            if os.path.isfile(kernel_root + '/' + file[:-1] + 'cc'):
                process(file[:-1] + 'cc')
            if os.path.isfile(kernel_root + '/' + file[:-2] + '_cpu.h'):
                process(file[:-2] + '_cpu.h')
            if os.path.isfile(kernel_root + '/' + file[:-2] + '_cpu.cc'):
                process(file[:-2] + '_cpu.cc')

    for file in os.listdir(kernel_root):
        process(file, True)

    for file in additional_src:
        process(file, False)

    for file in default_src:
        process(file, False)

    print('filegroup(')
    print('    name = "{0}",'.format(filegroup_name))
    print('    srcs = [')
    for src in sorted(set(srcs)):
        print('        "//tensorflow/core/kernels:{0}",'.format(src))
    print('    ],')
    print(')')


if __name__ == '__main__':
    main()
