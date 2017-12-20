from __future__ import division, print_function

from phyclone.tree import MarginalNode, Tree


class Particle(object):
    __slots__ = 'log_w', 'parent_particle', 'state'

    def __init__(self, log_w, parent_particle, state):
        self.log_w = log_w

        self.parent_particle = parent_particle

        self.state = state

    def copy(self):
        return Particle(self.log_w, self.parent_particle, self.state.copy())


class State(object):
    """ A partial state of the SMC algorithm.

    This class stores the partially constructed tree during the SMC.
    """

    def __init__(self, grid_size, node_idx, outliers, roots, alpha=1.0, outlier_prob=1e-4):

        self.node_idx = node_idx

        self.outliers = outliers

        self.roots = roots

        nodes = []

        for root in roots.values():
            nodes.extend(Tree.get_nodes(root))

        self.tree = Tree(grid_size, nodes, outliers, alpha=alpha, outlier_prob=outlier_prob)

    def __key(self):
        return (self.node_idx, self.root_idxs)

    def __eq__(self, y):
        return self.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def log_p(self):
        """ Log joint probability of the state marginalizing over value of dummy root.
        """
        return self.tree.log_p

    @property
    def log_p_one(self):
        """ Log joint probability of the state with dummy root having value of one.
        """
        return self.tree.log_p_one

    @property
    def root_idxs(self):
        return frozenset([x.idx for x in self.roots.values()])

    @property
    def root_nodes(self):
        return list(self.roots.values())


class Kernel(object):
    """ Abstract class representing an SMC kernel targeting the marginal FS-CRP distribution.

    Sub-classes should implement the get_proposal_distribution method.
    """

    def get_proposal_distribution(self, data_point, parent_particle):
        """ Get proposal distribution given the current data point and parent particle.
        """
        raise NotImplementedError

    def __init__(self, alpha, grid_size, outlier_prob):
        """
        Parameters
        ----------
        alpha: float
            Concentration parameter of the CRP.
        grid_size: int
            The size of the grid to approximate the recursion integrals.
        outlier_prob: float
            The prior probability a data point will not fit the tree.
        """
        self.alpha = alpha

        self.grid_size = grid_size

        self.outlier_prob = outlier_prob

    def create_particle(self, data_point, log_q, parent_particle, state):
        """  Create a new particle from a parent particle.
        """
        if parent_particle is None:
            log_w = state.log_p - log_q

        else:
            log_w = state.log_p - parent_particle.state.log_p - log_q

        return Particle(log_w, parent_particle, state)

    def create_state(self, data_point, parent_particle, node_idx, root_idxs):
        """ Create a new state.

        Parameters
        ----------
        data_point: array_like (float)
            Current data point.
        parent_particle: Particle
            Parent particle in genealogy.
        node_idx: int
            Index of the node the data point is assigned to.
        root_idxs: array_like (int)
            List of indexes for concrete nodes.
        """
        if parent_particle is None:
            outliers = []

            roots = {}

            if node_idx == -1:
                outliers.append(data_point)

            else:
                assert node_idx == 0

                assert root_idxs == set([0, ])

                roots[0] = MarginalNode(node_idx, self.grid_size)

                roots[node_idx].add_data_point(data_point)

        else:
            outliers = list(parent_particle.state.outliers)

            if node_idx == -1:
                outliers.append(data_point)

                roots = parent_particle.state.roots.copy()

            elif node_idx in parent_particle.state.roots:
                assert root_idxs == parent_particle.state.root_idxs

                roots = parent_particle.state.roots.copy()

                roots[node_idx] = roots[node_idx].copy(deep=False)

                roots[node_idx].add_data_point(data_point)

            else:
                child_idxs = parent_particle.state.root_idxs - root_idxs

                children = [parent_particle.state.roots[idx] for idx in child_idxs]

                roots = {}

                for idx in root_idxs:
                    if idx in parent_particle.state.root_idxs:
                        roots[idx] = parent_particle.state.roots[idx]

                roots[node_idx] = MarginalNode(node_idx, self.grid_size, children=children)

                roots[node_idx].add_data_point(data_point)

        return State(self.grid_size, node_idx, outliers, roots, alpha=self.alpha, outlier_prob=self.outlier_prob)

    def propose_particle(self, data_point, parent_particle):
        """ Propose a particle for t given a particle from t - 1 and a data point.
        """
        proposal_dist = self.get_proposal_distribution(data_point, parent_particle)

        state = proposal_dist.sample()

        log_q = proposal_dist.log_p(state)

        return self.create_particle(data_point, log_q, parent_particle, state)


class ProposalDistribution(object):
    """ Abstract class for proposal distribution.
    """

    def log_p(self, state):
        """ Get the log probability of the state.
        """
        raise NotImplementedError

    def sample(self):
        """ Sample a new state from the proposal distribution.
        """
        raise NotImplementedError
