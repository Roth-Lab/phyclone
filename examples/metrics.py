from phyclone.tree.utils import get_clades


def partition_metric(tree_1, tree_2):
    c1 = get_clades(tree_1)
    c2 = get_clades(tree_2)

    return len(c1.symmetric_difference(c2))
