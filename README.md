PhyClone
=========
Accurate Bayesian reconstruction of cancer phylogenies from bulk sequencing.
An implementation of the forest structured Chinese restaurant process with a Dirichlet prior on the node parameters.

--------

## Overview
1. [PhyClone Installation](#installation)
2. [Input File Formats](#input-files)
   * [Main input format](#main-input-format)
   * [Cluster input format](#cluster-file-format)
3. [Running PhyClone: Basic Usage](#running-phyclone)
   * [Outlier modeling options](#outlier-modelling)
4. [PhyClone Output](#phyclone-output)
   * [MAP Tree](#map-point-estimate-tree)
   * [Consensus Tree](#consensus-point-estimate-tree)
   * [Topology Report](#topology-report-and-sampled-topologies-archive)
-------

## Installation

PhyClone is currently in development so the following procedure has a few steps.

1. Ensure you have a working `conda` or (preferably) `mamba` installation.
You can do this by installing [Miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)


2. Install the required dependencies using mamba/conda.
We will create a new `conda` environment with the dependencies. Download the [environment.yaml](environment.yaml) file, and navigate into its save location. 
Run the following command:
```
mamba env create --file environment.yaml
```

3. Activate the `conda` environment.
```
mamba activate phyclone
```
> [!NOTE]
> You will have to do this whenever you open a new terminal and want to run PhyClone. 

4. Install PhyClone
```
pip install git+https://github.com/Roth-Lab/phyclone.git
```

Or, SSH command:

```
pip install git+ssh://git@github.com/Roth-Lab/phyclone.git
```

5. If everything worked PhyClone should be available on the command line.
```
phyclone --help
```

## Input File Formats

PhyClone analysis has two possible input files:
- [Main input file](#main-input-format) (**Required**)
- [Cluster file](#cluster-file-format)

> [!CAUTION]
> In principle PhyClone can be used without pre-clustering. 
> However, it drastically increases the computational complexity. Thus, pre-clustering is recommended for WGS data.

---------
### Main input format

To run a PhyClone analysis you will need to prepare an input file.
The file should be in tab delimited tidy data frame format and have the following columns.

> [!TIP]
> There is an example file in [examples/data/mixing.tsv](examples/data/mixing.tsv)

1. mutation_id - Unique identifier for the mutation. 
This is free form but should match across all samples.

> [!WARNING]
> PhyClone will remove any mutations without entries for all detected samples.
If you have mutations with no data in some samples set their counts to 0.

2. sample_id - Unique identifier for the sample.

3. ref_counts - Number of reads matching the reference allele.

4. alt_counts - Number of reads matching the alternate allele.

5. major_cn - Major copy number of segment overlapping mutation.

6. minor_cn - Minor copy number of segment overlapping mutation.

7. normal_cn - Total copy number of segment in healthy tissue.
For autosome this will be two and male sex chromosome one.

You can include the following optional columns:

1. tumour_content - The tumour content (cellularity) of the sample.
Default value is 1.0 if column is not present.
> [!NOTE]
> In principle this could be different for each mutation/sample.
However, in most cases it should be the same for all mutations in a sample.

2. error_rate - Sequencing error rate.
Default value is 0.001 if column is not present. 

------------------

### Cluster file format

[//]: # (> [!IMPORTANT])

[//]: # (> Though not strictly required to run PhyClone, this file is **strongly recommended**.)

> [!TIP]
> While any mutation pre-clustering method can be used, we recommend 
> [PyClone-VI](https://github.com/Roth-Lab/pyclone-vi). Both due to its established 
> strong performance, and its output format which can be fed directly into PhyClone *'as-is'*.

The file should be in tab delimited tidy data frame format and have the following columns.

1. mutation_id - Unique identifier for the mutation. 

    This is free form but should match across all samples and **must** match the identifiers provided
    in the [main input file](#main-input-format).

2. sample_id - Unique identifier for the sample.
3. cluster_id - Cluster that the mutation has been assigned to.

You can include the following optional columns:

4. chrom - Chromosome on which mutation_id is found
5. ccf - Cluster cellular prevalence estimate (included in all [PyClone-VI](https://github.com/Roth-Lab/pyclone-vi) clustering results)

> [!NOTE] In order to make use of PhyClone's data informed loss probability prior assignment, columns 4 and 5 are required.

[//]: # (4. outlier_prob - &#40;Prior&#41; probability that the cluster/mutation is an outlier.)

[//]: # (    )
[//]: # (    Default value is made equal to the current analysis `--outlier-prob` option if column is not present. )

[//]: # ()
[//]: # (> [!NOTE])

[//]: # (> The `--outlier-prob` option carries a default value of 0.0)

> [!TIP]
> There is an example file in [examples/data/mixing_clusters.tsv](examples/data/mixing_clusters.tsv)

-----------------

## Running PhyClone

PhyClone analyses are broken into two parts. 
First, sampling is performed using the `run` sub-command.
Second, the output sampling trace from the sampling `run` can be summarised as either a point-estimate tree ([MAP](#map-point-estimate-tree) or [Consensus](#consensus-point-estimate-tree)) or topology report.

Sampling can be run as follows:
```
phyclone run -i INPUT.tsv -c CLUSTERS.tsv -o TRACE.pkl.gz --num-chains 4
``` 
Which will take the [`INPUT.tsv`](#main-input-format) and (optionally) the [`CLUSTERS.tsv`](#cluster-file-format) file, as described above and write the trace file `TRACE.pkl.gz` in a compressed Python pickle format.

Relevant program options:
* `--num-chains` command controls how many independent parallel PhyClone sampling chains to use. Though the default value is set to 1, PhyClone will benefit from running with at least 4 chains, if the compute cores can be spared.
* `-n` command can be used to control the number of iterations of sampling to perform.
* `-b` command can be used to control the number of burn-in iterations to perform.
* `--seed` command can be used to seed the random number generator for reproducible results.
* 
> [!NOTE]
> Burn-in is done using a heuristic strategy of unconditional SMC.
All samples from the burn-in are discarded as they will not target the posterior.
* The `-d` command can be used to select the emission density. 
  * As in PyClone, the `binomial` and `beta-binomial` densities are available.


For more advanced options, run:
```
phyclone run --help
``` 

[//]: # (> [!IMPORTANT])

[//]: # (> Unlike PyClone, PhyClone does not estimate the precision parameter of the Beta-Binomial. This parameter can be set with the --precision flag, though the default value of 400 should suffice in most cases.)

### Outlier Modelling

As explored in the PhyClone paper, PhyClone is equipped with the ability to model mutational outliers and loss. There are two main approaches to running PhyClone with outlier modelling:
1. Using a global outlier probability.
   * If running on un-clustered data, this is the only option available to activate outlier modelling. 
      * Use `--outlier-prob <user-defined-loss-prior-probability>` replacing the `<>` text with a decimal value in the [0, 1] range. Barring prior knowledge, 0.001 should suffice. 
   > [!NOTE] This option will also allow for the use of a global loss probability prior on clustered runs as well.
2. Assigning the outlier probability from clustered data.
   * PhyClone is also able to split clusters into either high-loss or low-loss probability groupings. This feature requires that the clustered data include mutational chromosome assignments (which can be supplied in either the [data.tsv](#main-input-format) or [cluster.tsv](#cluster-file-format) files) and cluster cellular prevalence (CCF) measures.
   * To activate this feature, ensure the input files are populated with the appropriate columns and include the `--assign-loss-prob` flag in the PhyClone `run` command.
   > [!TIP] If using PyClone-VI for clustering, the CCF column will come as a part of its results. And you need only append the chromosomal positioning column `chrom` to either input files.
   
> [!IMPORTANT] With outlier modelling active, the end result table will assign all mutations inferred to be lost or outliers to a clone with the id of `-1`.

-----------------

## PhyClone Output

PhyClone includes ways three ways to summarise the results from a sampling trace file.
Two of which produce a point-estimate (a single tree), and a third which can reports on and can optionally build results for all uniquely sampled topologies:
1. [MAP tree](#map-point-estimate-tree)
   * **(Recommended)** Retrieves the tree with the highest sampled joint-likelihood.
2. [Consensus tree](#consensus-point-estimate-tree)
   * Produces a tree built from the consensus of clades across the entire sample trace.
3. [Topology report and archive](#topology-report-and-sampled-topologies-archive)
   * Produces a summary report table and (optionally) archive file of all uniquely sampled topologies from an analysis run.

### MAP Point-Estimate Tree

To build the PhyClone MAP tree, you can run the `map` command as follows:
```
phyclone map -i TRACE.pkl.gz -t TREE.nwk -o TABLE.tsv
``` 
Where `TRACE.pkl.gz` is the result from a PhyClone sampling run.

Expected output:
* `TREE.nwk` the inferred MAP clone tree topology in newick format. 
* `TABLE.tsv` a results table which contains: the assignment of mutations to clones, CCF (cellular prevalence) estimates, and clonal prevalence estimates per sample.

For more advanced options, run:
```
phyclone map --help
``` 

### Consensus Point-Estimate Tree

To build the PhyClone consensus tree, you can run the `consensus` command as follows:
```
phyclone consensus -i TRACE.pkl.gz -t TREE.nwk -o TABLE.tsv
``` 
Where `TRACE.pkl.gz` is the result from a PhyClone sampling run.

Expected output:
* `TREE.nwk` the inferred MAP clone tree topology in newick format. 
* `TABLE.tsv` a results table which contains: the assignment of mutations to clones, CCF (cellular prevalence) estimates, and clonal prevalence estimates per sample.

For more advanced options, run:
```
phyclone consensus --help
``` 

### Topology Report and Sampled Topologies Archive

Additionally, PhyClone is able to produce a summary report and archive file of all uniquely sampled topologies from an analysis run.
The 

To build the PhyClone topology report and full sampled topologies archive, run the `topology-report` command as follows:
```
phyclone topology-report -i TRACE.pkl.gz -o TOPOLOGY_TABLE.tsv -t SAMPLED_TOPOLOGIES.tar.gz
``` 
Where `TRACE.pkl.gz` is the result from a PhyClone sampling run.

Expected output:
* `TOPOLOGY_TABLE.tsv`, a high-level report table detailing each topology's log-likelihood, number of times sampled, and topology identifier (which can be
used to identify the tree in the accompanying topologies archive).
* `SAMPLED_TOPOLOGIES.tar.gz`, a compressed archive where each folder represents a uniquely sampled topology, folder names align with topology identifiers found in the `TOPOLOGY_TABLE.tsv`

Expected output, for each sampled topology folder in the `SAMPLED_TOPOLOGIES.tar.gz` (sampled-topologies archive):
* `TREE.nwk` the inferred MAP clone tree topology in newick format. 
* `TABLE.tsv` a results table which contains: the assignment of mutations to clones, CCF (cellular prevalence) estimates, and clonal prevalence estimates per sample.

Additional options:
* `--top-trees` can be used to define that only the top (user-defined-value) `x` trees should be built.
  * trees are ranked by their log-likelihood, such that the command `--top-trees 3`, would populate the archive with only the top 3 most likely trees.

# License

PhyClone is licensed under the GPL v3, see the [LICENSE](LICENSE.md) file for details.

# Versions

## 0.5.1

