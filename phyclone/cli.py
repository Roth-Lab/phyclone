import click

import phyclone.run


@click.command(
    context_settings={"max_content_width": 120}
)
@click.option(
    "-i", "--in-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path to trace file from MCMC analysis. Format is gzip compressed Python pickle file."""
)
@click.option(
    "-o", "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True)
)
@click.option(
    "-t", "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True)
)
@click.option(
    "-p", "--out-log-probs-file",
    # required=True,
    default=None,
    type=click.Path(resolve_path=True)
)
@click.option(
    "--consensus-threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    help="""Consensus threshold to keep an SNV."""
)
def consensus(**kwargs):
    """ Build consensus results.
    """
    phyclone.run.write_consensus_results(**kwargs)


@click.command(
    context_settings={"max_content_width": 120}
)
@click.option(
    "-i", "--in-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path to trace file from MCMC analysis. Format is gzip compressed Python pickle file."""
)
@click.option(
    "-o", "--out-table-file",
    required=True,
    type=click.Path(resolve_path=True)
)
@click.option(
    "-t", "--out-tree-file",
    required=True,
    type=click.Path(resolve_path=True)
)
@click.option(
    "-p", "--out-log-probs-file",
    default=None,
    type=click.Path(resolve_path=True)
)
@click.option(
    "--topology-report/--no-topology-report",
    default=False,
    help="Whether the topology counts should be computed."
)
def map(**kwargs):
    """ Build MAP results.
    """
    phyclone.run.write_map_results(**kwargs)


# =========================================================================
# Analysis
# =========================================================================
@click.command(
    context_settings={"max_content_width": 120},
    name="run"
)
@click.option(
    "-i", "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to TSV format file with copy number and allele count information for all samples. See the examples directory in the GitHub repository for format."""
)
@click.option(
    "-o", "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path to where trace file will be written in gzip compressed pickle format."""
)
@click.option(
    "-b", "--burnin",
    default=1,
    type=int,
    help="""Number of burnin iterations using unconditional SMC sampler. Default is 1."""
)
@click.option(
    "-n", "--num-iters",
    default=1000,
    type=int,
    help="""Number of iterations of the MCMC sampler to perform. Default is 1,000."""
)
@click.option(
    "-t", "--thin",
    default=1,
    type=int,
    help="""Thinning parameter for storing entries in trace. Default is 1."""
)
@click.option(
    "--num-threads",
    default=1,
    type=int,
    help="""Number of parallel threads for sampling. Default is 1."""
)
@click.option(
    "-c", "--cluster-file",
    default=None,
    type=click.Path(resolve_path=True),
    help="""Path to file with pre-computed cluster assignments of mutations is located."""
)
@click.option(
    "-d", "--density",
    default="beta-binomial",
    type=click.Choice(["binomial", "beta-binomial"]),
    help="""Allele count density in the PyClone model. Use beta-binomial for most cases. Default beta-binomial."""
)
@click.option(
    "-l", "--outlier-prob",
    default=0,
    type=float,
    help="""Prior probability data points are outliers and don't fit tree. Default is 0.0"""
)
@click.option(
    "-p", "--proposal",
    default="fully-adapted",
    type=click.Choice(["bootstrap", "fully-adapted", "semi-adapted"]),
    help="""
    Proposal distribution to use for PG sampling.
    Fully adapted is the most computationally expensive but also likely to lead to the best performance per iteration.
    For large datasets it may be necessary to use one of the other proposals.
    Default is fully-adapted.
    """
)
@click.option(
    "-s",
    "--subtree-update-prob",
    default=0.0,
    type=float,
    help="""Probability of updating a subtree (instead of whole tree) using PG sampler. Default is 0.0"""
)
@click.option(
    "-t",
    "--max-time",
    default=float("inf"),
    type=float,
    help="""Maximum running time in seconds."""
)
@click.option(
    "--concentration-update/--no-concentration-update",
    default=True,
    help="Whether the concentration parameter should be updated during sampling."
)
@click.option(
    "--mitochondrial/--not-mitochondrial",
    default=False,
    help="Whether the analysis is mitochondrial or not."
)
@click.option(
    "--concentration-value",
    default=1.0,
    type=float,
    help="""The (initial) concentration of the Dirichlet process. Higher values will encourage more clusters, lower values have the opposite effect. Default is 1.0."""
)
@click.option(
    "--grid-size",
    default=101,
    type=int,
    help="""Grid size for discrete approximation. This will numerically marginalise the cancer cell fraction. Higher values lead to more accurate approximations at the expense of run time."""
)
@click.option(
    "--num-particles",
    default=20,
    type=int,
    help="""Number of particles to use during PG sampling. Default is 20."""
)
@click.option(
    "--num-samples-data-point",
    default=1,
    type=int,
    help="""Number of Gibbs updates to reassign data points per SMC iteration. Default is 1."""
)
@click.option(
    "--num-samples-prune-regraph",
    default=1,
    type=int,
    help="""Number of prune-regraph updates per SMC iteration. Default is 1."""
)
@click.option(
    "--precision",
    default=400,
    type=float,
    help="""The (initial) precision parameter of the Beta-Binomial density. The higher the value the more similar the Beta-Binomial is to a Binomial. Default is 400."""
)
@click.option(
    "--print-freq",
    default=10,
    type=int,
    help="""How frequently to print information about fitting. Default every 10 iterations."""
)
@click.option(
    "--resample-threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0, clamp=True),
    help="""ESS threshold to trigger resampling. Default is 0.5."""
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="""Set random seed so results can be reproduced. By default a random seed is chosen."""
)
def run(**kwargs):
    """ Run a new PhyClone analysis.
    """
    phyclone.run.run(**kwargs)


# =========================================================================
# Setup main interface
# =========================================================================
@click.group(name="phyclone")
def main():
    pass


main.add_command(consensus)
main.add_command(map)
main.add_command(run)
