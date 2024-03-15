from collections import OrderedDict, defaultdict

import numba
import numpy as np
import pandas as pd

import phyclone.data.base
import phyclone.utils.math
from phyclone.utils.exceptions import MajorCopyNumberError


def load_data(file_name, cluster_file=None, density='beta-binomial', grid_size=101, outlier_prob=1e-4, precision=400):
    pyclone_data, samples, num_mutations = load_pyclone_data(file_name)

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
        cluster_df = pd.read_csv(cluster_file, sep="\t")

        if 'outlier_prob' not in cluster_df.columns:
            print('Cluster level outlier probability column not found. Setting values to {p}'.format(p=outlier_prob))
            cluster_df.loc[:, 'outlier_prob'] = outlier_prob

        if outlier_prob == 0:
            cluster_df.loc[:, 'outlier_prob'] = outlier_prob

        cluster_df = cluster_df[["mutation_id", "cluster_id", "outlier_prob"]].drop_duplicates()
        
        cluster_sizes = cluster_df["cluster_id"].value_counts().to_dict()

        clusters = cluster_df.set_index("mutation_id")["cluster_id"].to_dict()

        cluster_outlier_probs = cluster_df.set_index("cluster_id")["outlier_prob"].to_dict()
        
        print("Using input clustering with {} clusters".format(cluster_df["cluster_id"].nunique()))

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

    return data, samples, num_mutations


def compute_outlier_prob(outlier_prob, cluster_size):
    if outlier_prob == 0:
        return outlier_prob, np.log(1.0)
    else:
        res = np.log(outlier_prob) * cluster_size
        res_not = np.log1p(-outlier_prob) * cluster_size
        return res, res_not


def load_pyclone_data(file_name):
    df = pd.read_table(file_name)

    if len(df.columns) == 1:
        df = pd.read_csv(file_name)

    df['sample_id'] = df['sample_id'].astype(str)

    samples = sorted(df['sample_id'].unique())

    # Filter for mutations present in all samples
    df = df.groupby(by='mutation_id').filter(lambda x: sorted(x['sample_id'].unique()) == samples)

    #TODO: break the following into a fxn

    depth_est = df['alt_counts'].mean() + df['ref_counts'].mean()

    mutations = sorted(df['mutation_id'].unique())

    if len(samples) > 10:
        print('Num Samples: {}'.format(len(samples)))
        print('Samples: {}...'.format(' '.join(samples[:4])))
    else:
        print('Samples: {}'.format(' '.join(samples)))

    num_mutations = len(mutations)

    print('Num mutations: {}'.format(num_mutations))

    if 'error_rate' not in df.columns:
        df.loc[:, 'error_rate'] = 1e-3

    if 'tumour_content' not in df.columns:
        print('Tumour content column not found. Setting values to 1.0.')

        df.loc[:, 'tumour_content'] = 1.0

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

    return data, samples, num_mutations


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

    log_pi = phyclone.utils.math_utils.log_normalize(np.array(log_pi, dtype=float))

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
    ('cn', numba.int64[:,:]),
    ('mu', numba.float64[:,:]),
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

        ll[c] = data.log_pi[c] + phyclone.utils.math_utils.log_beta_binomial_pdf(data.a + data.b, data.b, a, b)

    return phyclone.utils.math_utils.log_sum_exp(ll)


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

        ll[c] = data.log_pi[c] + phyclone.utils.math_utils.log_binomial_pdf(data.a + data.b, data.b, e_vaf)

    return phyclone.utils.math_utils.log_sum_exp(ll)
