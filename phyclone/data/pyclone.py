import itertools
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numba
import numpy as np
import pandas as pd

import phyclone.data.base
from phyclone.utils.math import log_normalize, log_beta_binomial_pdf, log_sum_exp, log_binomial_pdf
from phyclone.utils.exceptions import MajorCopyNumberError
from dataclasses import dataclass


def load_data(file_name, rng, low_loss_prob, high_loss_prob, assign_loss_prob, cluster_file=None,
              density='beta-binomial', grid_size=101, outlier_prob=1e-4, precision=400):
    pyclone_data, samples = load_pyclone_data(file_name)

    if cluster_file is None:
        data = []

        for idx, (mut, val) in enumerate(pyclone_data.items()):
            out_probs = compute_outlier_prob(outlier_prob, 1)
            data_point = phyclone.data.base.DataPoint(idx,
                                                      val.to_likelihood_grid(density, grid_size, precision=precision),
                                                      name=mut, outlier_prob=out_probs[0],
                                                      outlier_prob_not=out_probs[1])

            data.append(data_point)

    else:
        cluster_df = _setup_cluster_df(cluster_file, file_name, outlier_prob,
                                       rng, low_loss_prob, high_loss_prob, assign_loss_prob)

        cluster_sizes = cluster_df["cluster_id"].value_counts().to_dict()

        clusters = cluster_df.set_index("mutation_id")["cluster_id"].to_dict()

        cluster_outlier_probs = cluster_df.set_index("cluster_id")["outlier_prob"].to_dict()

        print("Using input clustering with {} clusters".format(cluster_df["cluster_id"].nunique()))

        data = _create_clustered_data_arr(cluster_outlier_probs, cluster_sizes, clusters, density, grid_size,
                                          precision, pyclone_data)

    return data, samples


def _create_clustered_data_arr(cluster_outlier_probs, cluster_sizes, clusters, density, grid_size, precision,
                               pyclone_data):
    raw_data = defaultdict(list)
    for mut, val in pyclone_data.items():
        raw_data[clusters[mut]].append(val.to_likelihood_grid(density, grid_size, precision=precision))
    data = []
    for idx, cluster_id in enumerate(sorted(raw_data.keys())):
        val = np.sum(np.array(raw_data[cluster_id]), axis=0)
        cluster_outlier_prob = cluster_outlier_probs[cluster_id]
        out_probs = compute_outlier_prob(cluster_outlier_prob, cluster_sizes[cluster_id])

        data_point = phyclone.data.base.DataPoint(idx, val, name="{}".format(cluster_id),
                                                  outlier_prob=out_probs[0], outlier_prob_not=out_probs[1])

        data.append(data_point)
    return data


def _setup_cluster_df(cluster_file, data_file, outlier_prob, rng, low_loss_prob, high_loss_prob, assign_loss_prob):
    cluster_df = pd.read_csv(cluster_file, sep="\t")
    if 'outlier_prob' not in cluster_df.columns:
        if assign_loss_prob:
            if 'chrom' not in cluster_df.columns:
                data_df = pd.read_table(data_file)
                data_df = data_df[['mutation_id', 'chrom', 'coord']]
                cluster_df = pd.merge(cluster_df, data_df, how="inner", on=["mutation_id"])
                cluster_df = cluster_df.drop_duplicates()
            print('Cluster level outlier probability column not found. Assigning from data.')
            _assign_out_prob(cluster_df, rng, low_loss_prob, high_loss_prob)
        else:
            print('Cluster level outlier probability column not found. Setting values to {p}'.format(p=outlier_prob))
            cluster_df.loc[:, 'outlier_prob'] = outlier_prob
    if outlier_prob == 0 and not assign_loss_prob:
        cluster_df.loc[:, 'outlier_prob'] = outlier_prob
    else:
        cluster_df.loc[cluster_df['outlier_prob'] == 0, 'outlier_prob'] = outlier_prob
    cluster_df = cluster_df[["mutation_id", "cluster_id", "outlier_prob"]].drop_duplicates()
    return cluster_df


def _assign_out_prob(df, rng, low_loss_prob, high_loss_prob):

    truncal_cluster = _define_truncal_cluster(df)

    cluster_info_dict = _build_cluster_info_dict(df)

    truncal_dists = _get_truncal_chrom_arr(df, truncal_cluster)

    lost_clusters = _define_possibly_lost_clusters(cluster_info_dict, rng, truncal_cluster, truncal_dists)

    _finalize_loss_prob_on_cluster_df(df, high_loss_prob, lost_clusters, low_loss_prob)


def _define_possibly_lost_clusters(cluster_info_dict, rng, truncal_cluster, truncal_dists):
    truncal_dist_len = len(truncal_dists)
    lost_clusters = list()
    min_clust_size = 4
    test_iters = 10000
    for cluster, info_obj in cluster_info_dict.items():
        cluster_dist_len = info_obj.num_mutations
        if cluster == truncal_cluster or cluster_dist_len < min_clust_size:
            continue

        if cluster_dist_len > truncal_dist_len:
            truncal_dist_tester = np.resize(truncal_dists, cluster_dist_len)
        else:
            truncal_dist_tester = truncal_dists

        chromo_obvs = info_obj.num_unique_chromosomes

        samples_fewer = 0

        num_unique_sum = 0

        for i in range(test_iters):
            sample_drawn = rng.choice(truncal_dist_tester, size=cluster_dist_len, replace=False)
            unique_vals = np.unique(sample_drawn)
            num_unique = len(unique_vals)
            num_unique_sum += num_unique
            if num_unique < chromo_obvs:
                samples_fewer += 1

        if samples_fewer == 0:
            pvalue = 1 / test_iters
        else:
            pvalue = samples_fewer / test_iters

        estimate = (num_unique_sum / test_iters) / chromo_obvs

        if pvalue < 0.01 and estimate > 1:
            lost_clusters.append(cluster)
    return lost_clusters


def _finalize_loss_prob_on_cluster_df(cluster_df, high_loss_prob, lost_clusters, low_loss_prob):
    cluster_df['outlier_prob'] = low_loss_prob
    value_filter = cluster_df['cluster_id'].isin(lost_clusters)
    cluster_df.loc[value_filter, 'outlier_prob'] = high_loss_prob
    if len(lost_clusters) > 0:
        print("{} potentially lost/outlier clusters identified,"
              " setting their prior loss prob to {}.".format(len(lost_clusters), high_loss_prob))
        print("Clusters identified as potentially lost/outliers: {}".format(lost_clusters))
    else:
        print("No potentially lost/outlier clusters identified,"
              " setting global prior loss prob to {}.".format(low_loss_prob))


def _get_truncal_chrom_arr(df, truncal_cluster):
    df = df[['cluster_id', 'chrom', 'coord', 'mutation_id']].drop_duplicates()
    df = df.loc[df['cluster_id'] == truncal_cluster]
    truncal_dists = list()
    grouped = df.groupby('chrom', sort=False)
    for chrom, group in grouped:
        truncal_dists.extend([chrom] * len(group))
    truncal_dists = np.array(truncal_dists)
    return truncal_dists


def _define_truncal_cluster(df):
    grouped = df.groupby('cluster_id', sort=False)
    cluster_prev_dict = dict()
    for cluster, group in grouped:
        unique_vals = group['cellular_prevalence'].unique()
        sum_vals = unique_vals.mean()
        cluster_prev_dict[cluster] = sum_vals

    # TODO: check for multiple?
    truncal_cluster = max(cluster_prev_dict.items(), key=itemgetter(1))[0]
    print("Cluster {} identified as likely truncal.".format(truncal_cluster))
    return truncal_cluster


def _build_cluster_info_dict(df):
    grouped = df.groupby('cluster_id', sort=False)
    cluster_info_dict = dict()
    for cluster, group in grouped:
        clust_info_obj = ClusterInfo(cluster_id=cluster,
                                     num_mutations=len(group['mutation_id'].unique()),
                                     num_unique_chromosomes=len(group['chrom'].unique()))
        cluster_info_dict[cluster] = clust_info_obj
    return cluster_info_dict


@dataclass
class ClusterInfo:
    cluster_id: str | int
    num_mutations: int
    num_unique_chromosomes: int


def compute_outlier_prob(outlier_prob, cluster_size):
    if outlier_prob == 0:
        return outlier_prob, np.log(1.0)
    else:
        res = np.log(outlier_prob) * cluster_size
        res_not = np.log1p(-outlier_prob) * cluster_size
        return res, res_not


def load_pyclone_data(file_name):
    df = _create_raw_data_df(file_name)

    # remove any rows where maj copy number == 0
    df = df.loc[df['major_cn'] > 0]

    samples = sorted(df['sample_id'].unique())

    # Filter for mutations present in all samples
    df = df.groupby(by='mutation_id').filter(lambda x: sorted(x['sample_id'].unique()) == samples)

    mutations = sorted(df['mutation_id'].unique())

    print('Num mutations: {}'.format(len(mutations)))

    _process_required_cols_on_df(df, samples)

    data = _create_loaded_pyclone_data_dict(df, mutations, samples)

    return data, samples


def _process_required_cols_on_df(df, samples):
    if len(samples) > 10:
        print('Num Samples: {}'.format(len(samples)))
        print('Samples: {}...'.format(' '.join(samples[:4])))
    else:
        print('Samples: {}'.format(' '.join(samples)))
    if 'error_rate' not in df.columns:
        df.loc[:, 'error_rate'] = 1e-3
    if 'tumour_content' not in df.columns:
        print('Tumour content column not found. Setting values to 1.0.')

        df.loc[:, 'tumour_content'] = 1.0


def _create_raw_data_df(file_name):
    df = pd.read_table(file_name)
    if len(df.columns) == 1:
        df = pd.read_csv(file_name)
    df['sample_id'] = df['sample_id'].astype(str)
    return df


def _create_loaded_pyclone_data_dict(df, mutations, samples):
    data = OrderedDict()
    for name in mutations:
        mut_df = df[df['mutation_id'] == name]

        sample_data_points = []

        mut_df = mut_df.set_index('sample_id')

        for sample in samples:
            row = mut_df.loc[sample]

            a = row['ref_counts']

            b = row['alt_counts']

            cn, mu, log_pi = get_major_cn_prior(
                row['major_cn'],
                row['minor_cn'],
                row['normal_cn'],
                error_rate=row['error_rate']
            )

            sample_data_points.append(
                SampleDataPoint(a, b, cn, mu, log_pi, row['tumour_content'])
            )

        data[name] = DataPoint(samples, sample_data_points)
    return data


def get_major_cn_prior(major_cn, minor_cn, normal_cn, error_rate=1e-3):
    total_cn = major_cn + minor_cn

    cn = []

    mu = []

    log_pi = []

    if major_cn < minor_cn:
        raise MajorCopyNumberError(major_cn, minor_cn)

    # Consider all possible mutational genotypes consistent with mutation before CN change
    for x in range(1, major_cn + 1):
        cn.append((normal_cn, normal_cn, total_cn))

        mu.append((error_rate, error_rate, min(1 - error_rate, x / total_cn)))

        log_pi.append(0)

    # Consider mutational genotype of mutation before CN change if not already added
    mutation_after_cn = (normal_cn, total_cn, total_cn)

    if mutation_after_cn not in cn:
        cn.append(mutation_after_cn)

        mu.append((error_rate, error_rate, min(1 - error_rate, 1 / total_cn)))

        log_pi.append(0)

        assert len(set(cn)) == 2

    cn = np.array(cn, dtype=int)

    mu = np.array(mu, dtype=float)

    log_pi = log_normalize(np.array(log_pi, dtype=float))

    return cn, mu, log_pi


class DataPoint(object):

    def __init__(self, samples, sample_data_points):
        self.samples = samples

        self.sample_data_points = sample_data_points

    def get_ccf_grid(self, grid_size):
        return np.linspace(0, 1, grid_size)

    def to_dict(self):
        return OrderedDict(zip(self.samples, self.sample_data_points))

    def to_likelihood_grid(self, density, grid_size, precision=None):
        if (density == 'beta-binomial') and (precision is None):
            raise Exception('Precision must be set when using Beta-Binomial.')

        shape = (len(self.samples), grid_size)

        log_ll = np.zeros(shape)

        sample_data_points = self.sample_data_points
        ccf_grid = self.get_ccf_grid(grid_size)

        _compute_liklihood_grid(ccf_grid, density, log_ll, precision, numba.typed.List(sample_data_points))

        return log_ll


@numba.jit(cache=True, nopython=True)
def _compute_liklihood_grid(ccf_grid, density, log_ll, precision, sample_data_points):
    for s_idx, data_point in enumerate(sample_data_points):
        for i, ccf in enumerate(ccf_grid):
            if density == 'beta-binomial':
                log_ll[s_idx, i] = log_pyclone_beta_binomial_pdf(data_point, ccf, precision)

            elif density == 'binomial':
                log_ll[s_idx, i] = log_pyclone_binomial_pdf(data_point, ccf)


@numba.experimental.jitclass([
    ('a', numba.int64),
    ('b', numba.int64),
    ('cn', numba.int64[:, :]),
    ('mu', numba.float64[:, :]),
    ('log_pi', numba.float64[:]),
    ('t', numba.float64)
])
class SampleDataPoint(object):

    def __init__(self, a, b, cn, mu, log_pi, t):
        self.a = a
        self.b = b
        self.cn = cn
        self.mu = mu
        self.log_pi = log_pi
        self.t = t


@numba.jit(nopython=True)
def log_pyclone_beta_binomial_pdf(data, f, s):
    t = data.t

    C = len(data.cn)

    population_prior = np.zeros(3)
    population_prior[0] = (1 - t)
    population_prior[1] = t * (1 - f)
    population_prior[2] = t * f

    ll = np.ones(C, dtype=np.float64) * np.inf * -1

    for c in range(C):
        e_vaf = 0

        norm_const = 0

        for i in range(3):
            e_cn = population_prior[i] * data.cn[c, i]

            e_vaf += e_cn * data.mu[c, i]

            norm_const += e_cn

        e_vaf /= norm_const

        a = e_vaf * s

        b = s - a

        ll[c] = data.log_pi[c] + log_beta_binomial_pdf(data.a + data.b, data.b, a, b)

    return log_sum_exp(ll)


@numba.jit(nopython=True)
def log_pyclone_binomial_pdf(data, f):
    t = data.t

    C = len(data.cn)

    population_prior = np.zeros(3)
    population_prior[0] = (1 - t)
    population_prior[1] = t * (1 - f)
    population_prior[2] = t * f

    ll = np.ones(C, dtype=np.float64) * np.inf * -1

    for c in range(C):
        e_vaf = 0

        norm_const = 0

        for i in range(3):
            e_cn = population_prior[i] * data.cn[c, i]

            e_vaf += e_cn * data.mu[c, i]

            norm_const += e_cn

        e_vaf /= norm_const

        ll[c] = data.log_pi[c] + log_binomial_pdf(data.a + data.b, data.b, e_vaf)

    return log_sum_exp(ll)
