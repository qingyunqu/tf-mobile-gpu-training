import click

from . import load_graph


@click.command()
@click.option('--quantized', type=click.Path())
@click.option('--graph_def_out', type=click.Path())
def main(quantized, graph_def_out):
    g = load_graph(quantized)
    with open(graph_def_out, 'wb') as fp:
        fp.write(g.graph.SerializeToString())


if __name__ == '__main__':
    main()
