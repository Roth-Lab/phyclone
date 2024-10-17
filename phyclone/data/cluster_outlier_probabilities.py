from _operator import itemgetter
from collections import Counter
from dataclasses import dataclass
import numpy as np


def _assign_out_prob(df, rng, low_loss_prob, high_loss_prob):
    truncal_cluster = _define_truncal_cluster(df)

    print("Cluster {} identified as likely truncal.".format(truncal_cluster))

    cluster_info_dict = _build_cluster_info_dict(df)

    truncal_dists = _get_truncal_chrom_arr(df, truncal_cluster)

    lost_clusters = _define_possibly_lost_clusters(
        cluster_info_dict, rng, truncal_cluster, truncal_dists
    )

    _finalize_loss_prob_on_cluster_df(df, high_loss_prob, lost_clusters, low_loss_prob)


def _define_possibly_lost_clusters(
    cluster_info_dict, rng, truncal_cluster, truncal_dists
):
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
            sample_drawn = rng.choice(
                truncal_dist_tester, size=cluster_dist_len, replace=False
            )
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


def _finalize_loss_prob_on_cluster_df(
    cluster_df, high_loss_prob, lost_clusters, low_loss_prob
):
    cluster_df["outlier_prob"] = low_loss_prob
    value_filter = cluster_df["cluster_id"].isin(lost_clusters)
    cluster_df.loc[value_filter, "outlier_prob"] = high_loss_prob
    if len(lost_clusters) > 0:
        if len(lost_clusters) > 1:
            pluralize = "s"
            possessive = "their"
        else:
            pluralize = ""
            possessive = "its"
        print(
            "{num} potentially lost/outlier cluster{pl} identified,"
            " setting {pos} prior loss prob to {pr}.".format(
                num=len(lost_clusters), pl=pluralize, pr=high_loss_prob, pos=possessive
            )
        )
        print(
            "Cluster{pl} identified as potentially lost/outlier{pl}: {lost}".format(
                pl=pluralize, lost=lost_clusters
            )
        )
    else:
        print(
            "No potentially lost/outlier clusters identified,"
            " setting global prior loss prob to {}.".format(low_loss_prob)
        )


def _get_truncal_chrom_arr(df, truncal_cluster):
    df = df[["cluster_id", "chrom", "coord", "mutation_id"]].drop_duplicates()
    df = df.loc[df["cluster_id"] == truncal_cluster]
    truncal_dists = list()
    grouped = df.groupby("chrom", sort=False)
    for chrom, group in grouped:
        truncal_dists.extend([chrom] * len(group))
    truncal_dists = np.array(truncal_dists)
    return truncal_dists


def _define_truncal_cluster(df):
    potentials, unique_sample_names = _get_potential_truncal_clusters(df)

    counter_dict = Counter(potentials)
    freq_list = counter_dict.values()
    max_count = max(freq_list)

    if max_count == len(unique_sample_names):
        total = Counter(freq_list)[max_count]
        if total == 1:
            truncal_cluster = counter_dict.most_common(1)[0][0]
            return truncal_cluster

    potentials_set = set(potentials)

    df = df.loc[df["cluster_id"].isin(potentials_set)]

    grouped = df.groupby("cluster_id", sort=False)
    cluster_prev_dict = dict()
    for cluster, group in grouped:
        # TODO: check mean vs. median here
        sum_vals = group["cellular_prevalence"].mean()
        cluster_prev_dict[cluster] = sum_vals

    truncal_cluster = max(cluster_prev_dict.items(), key=itemgetter(1))[0]
    return truncal_cluster


def _get_potential_truncal_clusters(df):
    unique_sample_names = df["sample_id"].unique()
    grouped = df.groupby("sample_id", sort=False)
    potentials = list()
    for sample, group in grouped:
        group = group.sort_values(by="cellular_prevalence", ascending=False)
        top_prev = group["cellular_prevalence"].iloc[0]
        top_prev_group = group.loc[group["cellular_prevalence"] == top_prev]
        clusters_with_top_prev_in_sample = top_prev_group["cluster_id"].unique()
        potentials.extend(clusters_with_top_prev_in_sample)
    return potentials, unique_sample_names


def _build_cluster_info_dict(df):
    grouped = df.groupby("cluster_id", sort=False)
    cluster_info_dict = dict()
    for cluster, group in grouped:
        clust_info_obj = ClusterInfo(
            cluster_id=cluster,
            num_mutations=len(group["mutation_id"].unique()),
            num_unique_chromosomes=len(group["chrom"].unique()),
        )
        cluster_info_dict[cluster] = clust_info_obj
    return cluster_info_dict


@dataclass
class ClusterInfo:
    cluster_id: str | int
    num_mutations: int
    num_unique_chromosomes: int
