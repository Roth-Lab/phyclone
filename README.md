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

## Usage

### Input files

PhyClone analysis has two possible input files:
- [main input file](#main-input-format) (**Required**)
- [cluster file](#cluster-file-format)

> [!CAUTION]
> While PhyClone analysis can be minimally run with only the main input.tsv file, it is **strongly
recommended** to provide a cluster file as well.

---------
#### Main input format

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

You can include the following optional columns.

1. tumour_content - The tumour content (cellularity) of the sample.
Default value is 1.0 if column is not present.
> [!NOTE]
> In principle this could be different for each mutation/sample.
However, in most cases it should be the same for all mutations in a sample.

2. error_rate - Sequencing error rate.
Default value is 0.001 if column is not present. 

------------------

#### Cluster file format

> [!IMPORTANT]
> Though not strictly required to run PhyClone, this file is **strongly recommended**.

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

You can include the following optional column:

4. outlier_prob - (Prior) probability that the cluster/mutation is an outlier.
    
    Default value is made equal to the current analysis `--outlier-prob` option if column is not present. 

> [!NOTE]
> The `--outlier-prob` option carries a default value of 0.0

> [!TIP]
> There is an example file in [examples/data/mixing_clusters.tsv](examples/data/mixing_clusters.tsv)

-----------------

### Running PhyClone

PhyClone analyses are broken into two parts. 
First, sampling is performed using the `run` sub-command.
Second, a MAP tree is constructed from the trace using the `map` sub-command.

Sampling can be run as follows
```
phyclone run -i INPUT.tsv -c CLUSTERS.tsv -o TRACE.pkl 
``` 
which will take the `INPUT.tsv` and (optionally) the `CLUSTERS.tsv` file, as described above and write the trace file `TRACE.pkl` in a Python pickle format.

The `-n` command can be used to control the number of iterations of sampling to perform.

The `-b` command can be used to control the number of burnin iterations to perform.

> [!NOTE]
> Burnin is done using a heuristic strategy of unconditional SMC.
All samples from the burnin are discarded as they will not target the posterior.

The `-d` command can be used to select the emission density.
As in PyClone the `binomial` and `beta-binomial` densities are available.

> [!IMPORTANT]
> Unlike PyClone, PhyClone does not estimate the precision parameter of the Beta-Binomial.
This parameter can be set with the --precision flag.

To build the final results of PhyClone you can run the `map` command as follows.
```
phyclone map -i TRACE.pkl -t TREE.nwk -o TABLE.tsv
``` 
where `TRACE.pkl` is the result from the previous step, `TREE.nwk` is the output clone tree in newick format, and `TABLE.tsv` the assignment of mutations to clones.

# License

PhyClone is licensed under the GPL v3, see the [LICENSE](LICENSE.md) file for details.

# Versions

## 0.5.0

