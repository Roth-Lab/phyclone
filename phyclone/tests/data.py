import numpy as np
from scipy.stats import binom
from phyclone.data.pyclone import SampleDataPoint, get_major_cn_prior, DataPoint
from phyclone.data.base import DataPoint as DataPointFinal


def simulate_data(
        tree,
        rng,
        num_samples,
        depth=100,
        error_rate=1e-3,
        max_cn=6,
        min_cn=1,
        min_minor_cn=0,
        tumour_content=1.0,
        dist='beta-binomial',
        grid_size=101,
        precision=400,
        ):

    g_n = 'AA'

    samples = range(num_samples)

    for node in tree.nodes:
        samples_dict = {k:{} for k in range(num_samples)}
        tree.nodes[node]['converted_sample_dp'] = samples_dict
        node_sample_dp = []
        for snv in tree.nodes[node]['snvs']:

            node_cellular_prev_arr = tree.nodes[node]['cellular_prev']
            snv_datapoints = []
            for dim_idx, f in enumerate(node_cellular_prev_arr):
                minor_cn, major_cn = get_parental_copy_number(min_cn, max_cn, rng, min_minor_cn=min_minor_cn)

                ref_genotypes = get_reference_genotype(g_n, minor_cn, major_cn)

                g_r = rng.choice(ref_genotypes)

                var_genotypes = get_variant_genotypes(g_n, g_r, minor_cn, major_cn)

                g_v = rng.choice(var_genotypes)

                ref_counts, var_counts, mu = simulate_counts(
                    f,
                    tumour_content,
                    depth,
                    error_rate,
                    g_n,
                    g_r,
                    g_v,
                    rng
                )

                cn, mu_inf, log_pi = get_major_cn_prior(major_cn, minor_cn, len(g_n),error_rate,)

                sample_data_point = SampleDataPoint(ref_counts, var_counts, cn, mu_inf, log_pi, tumour_content)

                tree.nodes[node]['converted_sample_dp'][dim_idx][snv] = sample_data_point
                snv_datapoints.append(sample_data_point)

            node_sample_dp.append(DataPoint(samples, snv_datapoints).to_likelihood_grid(dist, grid_size, precision))
        node_vals = np.array(node_sample_dp)
        node_val = node_vals.sum(axis=0)
        tree.nodes[node]["datapoint"] = [DataPointFinal(node, node_val, outlier_prob=0.0, outlier_prob_not=np.log(1.0))]


def get_parental_copy_number(min_cn, max_cn, rng, min_minor_cn=0):
    total_cn = rng.integers(min_cn, max_cn + 1)

    x = rng.integers(0, total_cn)

    minor_cn = max(min_minor_cn, min(total_cn - x, x))

    major_cn = total_cn - minor_cn

    return minor_cn, major_cn


def get_reference_genotype(g_n, minor_cn, major_cn):
    total_cn = minor_cn + major_cn

    return [g_n, 'A' * total_cn]


def get_variant_genotypes(g_n, g_r, minor_cn, major_cn):
    # Mutation occurs before CN event
    if g_n == g_r:
        variant_genotypes = ['A' * minor_cn + 'B' * major_cn]

        if minor_cn > 0:
            variant_genotypes.append('A' * major_cn + 'B' * minor_cn)

    # Mutation occurs after CN event
    else:
        total_cn = minor_cn + major_cn

        variant_genotypes = ['A' * (total_cn - 1) + 'B']

    return variant_genotypes


def simulate_counts(f, t, depth, error_rate, g_n, g_r, g_v, rng):
    cn_n = _get_cn(g_n)
    cn_r = _get_cn(g_r)
    cn_v = _get_cn(g_v)

    mu_n = _get_mu(g_n, error_rate)
    mu_r = _get_mu(g_r, error_rate)
    mu_v = _get_mu(g_v, error_rate)

    p_n = (1 - t) * cn_n
    p_r = t * (1 - f) * cn_r
    p_v = t * f * cn_v

    norm_const = p_n + p_r + p_v

    p_n /= norm_const
    p_r /= norm_const
    p_v /= norm_const

    mu = p_n * mu_n + p_r * mu_r + p_v * mu_v

    var_counts = binom.rvs(depth, mu, random_state=rng)

    ref_counts = depth - var_counts

    return ref_counts, var_counts, mu


def _get_cn(g):
    return len(g)


def _get_mu(g, error_rate):
    num_ref_alleles = g.count('A')

    num_var_alleles = g.count('B')

    if num_ref_alleles == 0:
        return 1 - error_rate

    elif num_var_alleles == 0:
        return error_rate

    else:
        return num_var_alleles / (num_ref_alleles + num_var_alleles)
