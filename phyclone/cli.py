import click

import phyclone.run


@click.command(
    context_settings={'max_content_width': 120}
)
@click.option(
    '-d', '--data-file',
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help='''Path to TSV format file with copy number and allele count information for all samples. See the examples
    directory in the GitHub repository for format.'''
)
@click.option(
    '-t', '--trace-file',
    required=True,
    type=click.Path(resolve_path=True),
    help='''Path to trace file from MCMC analysis.'''
)
@click.option(
    '-o', '--out-table-file',
    required=True,
    type=click.Path(resolve_path=True)
)
@click.option(
    '-n', '--out-tree-file',
    required=True,
    type=click.Path(resolve_path=True)
)
def consensus(data_file, trace_file, out_table_file, out_tree_file):
    """ Build consensus result.
    """
    phyclone.run.post_process(data_file, trace_file, out_table_file, out_tree_file)


#=========================================================================
# Analysis
#=========================================================================
@click.command(
    context_settings={'max_content_width': 120},
    name='run'
)
@click.option(
    '-i', '--in-file',
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help='''Path to TSV format file with copy number and allele count information for all samples. See the examples
    directory in the GitHub repository for format.'''
)
@click.option(
    '-t', '--trace-file',
    required=True,
    type=click.Path(resolve_path=True),
    help='''Path to where trace file will be written in HDF5 format.'''
)
@click.option(
    '-b', '--burnin',
    default=100,
    type=int,
    help='''Number of burnin iterations using unconditional SMC sampler.'''
)
@click.option(
    '-d', '--density',
    default='beta-binomial',
    type=click.Choice(['binomial', 'beta-binomial']),
    help='''Allele count density in the PyClone model. Use beta-binomial for most cases. Default beta-binomial.'''
)
@click.option(
    '-n', '--num-iters',
    default=1000,
    type=int,
    help='''Number of iterations of the MCMC sampler to perform. Default is 10,000.'''
)
@click.option(
    '--concentration-value',
    default=1.0,
    type=float,
    help='''The (initial) concentration of the Dirichlet process. Higher values will encourage more clusters, lower
    values have the opposite effect. Default is 1.0.'''
)
@click.option(
    '--grid-size',
    default=101,
    type=int,
    help='''Grid size for discrete approximation. This will numerically marginalise the cancer cell fraction. Higher
    values lead to more accurate approximations at the expense of run time.'''
)
@click.option(
    '--precision',
    default=400,
    type=float,
    help='''The (initial) precision parameter of the Beta-Binomial density. The higher the value the more similar the
    Beta-Binomial is to a Binomial. Default is 400.'''
)
@click.option(
    '--seed',
    default=None,
    type=int,
    help='''Set random seed so results can be reproduced. By default a random seed is chosen.'''
)
def run(**kwargs):
    """ Run a new PhyClone analysis.
    """
    phyclone.run.run(**kwargs)


#=========================================================================
# Setup main interface
#=========================================================================
@click.group(name='phyclone')
def main():
    pass


main.add_command(consensus)
main.add_command(run)
