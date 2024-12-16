from typing import Union

import numpy as np

from phyclone.tree.utils import compute_log_S


class TreeNode(object):
    __slots__ = ("log_p", "log_r", "node_id", "data_points")

    def __init__(self, grid_size: tuple[int, int], log_prior: float, node_id: Union[str | int]):
        self.log_p = np.full(grid_size, log_prior, order="C")
        self.log_r = np.zeros(grid_size, order="C")
        self.node_id = node_id
        self.data_points = set()

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.log_p = self.log_p.copy()
        new.log_r = self.log_r.copy()
        if isinstance(self.node_id, str):
            new.node_id = str(self.node_id)
        else:
            new.node_id = int(self.node_id)
        new.data_points = self.data_points.copy()
        return new

    def __eq__(self, other):
        log_p_compare = np.array_equal(self.log_p, other.log_p)
        log_r_compare = np.array_equal(self.log_r, other.log_r)
        return log_p_compare and log_r_compare

    def add_data_point_list(self, data_point_list):
        dp_idx_set = {dp.idx for dp in data_point_list}
        assert self.data_points.isdisjoint(dp_idx_set)
        self.data_points.update(dp_idx_set)

        log_p = self.log_p
        log_r = self.log_r

        for data_point in data_point_list:
            log_p += data_point.value
            log_r += data_point.value

    def add_data_point(self, data_point):
        dp_idx = data_point.idx
        assert dp_idx not in self.data_points

        self.data_points.add(dp_idx)
        self.log_p += data_point.value
        self.log_r += data_point.value

    def remove_data_point(self, data_point):
        dp_idx = data_point.idx
        assert dp_idx in self.data_points

        self.data_points.discard(dp_idx)
        self.log_p -= data_point.value

    def update_node_from_child_r_vals(self, child_log_r_values):
        log_p = self.log_p
        log_r = self.log_r

        if len(child_log_r_values) == 0:
            np.copyto(log_r, log_p)
            return
        else:
            log_s = compute_log_S(child_log_r_values)

        np.add(log_p, log_s, out=log_r, order="C")

    def copy(self):
        return self.__copy__()

    def to_dict(self):
        return {"log_p": self.log_p, "log_R": self.log_r, "node_id": self.node_id}

    def serialize(self):
        return {"node_id": str(self.node_id), "data_points": str(list(self.data_points))}
