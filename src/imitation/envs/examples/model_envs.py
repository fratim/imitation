"""Example discrete MDPs for use with tabular MCE IRL."""

from typing import Optional

import gym
import numpy as np

from imitation.envs.resettable_env import TabularModelEnv


class GridWorld(TabularModelEnv):
    """A gridworld with objects"""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        horizon: int,
        rew_default: int = -1,
        rew_goal: int = 10,
        n_objects: int = 0,
        reduce_rows: bool = False,
        reduce_cols: bool = False,
    ):
        super().__init__()

        self.n_objects = n_objects
        self.n_levels = self.n_objects + 1
        self.reduce_rows = reduce_rows
        self.reduce_cols = reduce_cols

        self.width = width if not self.reduce_cols else 1
        self.height = height if not self.reduce_rows else 1

        self.reward_goal = rew_goal
        self.reward_default = rew_default

        object_positions = [[1, 2], [3, 3], [1, 6]]
        for obj in object_positions:
            if self.reduce_rows:
                obj[0] = 0
            if self.reduce_cols:
                obj[1] = 0

        if self.n_objects == 0:
            self.object_positions = []
        else:
            self.object_positions = object_positions[:self.n_objects]

        n_states = self.width * self.height * self.n_levels
        O_mat = self._observation_matrix = np.zeros((n_states, n_states), dtype=np.float32)

        n_actions = 4 + self.n_levels

        R_vec = self._reward_matrix = np.zeros((n_states,))
        T_mat = self._transition_matrix = np.zeros((n_states, n_actions, n_states))
        self._horizon = horizon

        for row in range(self.height):
            for col in range(self.width):
                for level in range(self.n_levels):
                    state_id = self.to_id_clamp(row, col, level)

                    # start by computing reward
                    R_vec[state_id] = self.get_reward(row, col, level)

                    # now compute observation
                    O_mat[state_id, state_id] = 1

                    # finally, compute transition matrix entries for each of the
                    T_mat = self.set_transition_matrix(row, col, level, state_id, T_mat)

        assert np.allclose(np.sum(T_mat, axis=-1), 1, rtol=1e-5)

    def set_transition_matrix(self, row, col, level, state_id, T_mat):
        for drow in [-1, 1]:
            for dcol in [-1, 1]:
                action_id = (drow + 1) + (dcol + 1) // 2
                if level == 0:
                    target_state = self.to_id_clamp(row + drow, col + dcol, level)
                    T_mat[state_id, action_id, target_state] += 1
                else:
                    T_mat[state_id, action_id, state_id] += 1

        for toggle in range(self.n_levels):
            action_id = 4 + toggle
            target_state = self.to_id_clamp(row, col, toggle)
            if toggle == 0:  # toggling zero always leads back to zero level
                T_mat[state_id, action_id, target_state] += 1
            elif level == 0 and ([row, col] == self.object_positions[toggle - 1]):  # correct position to toggle object
                T_mat[state_id, action_id, target_state] += 1
            else:  # not in correct position to toggle object so nothing happens
                T_mat[state_id, action_id, state_id] += 1

        return T_mat

    def get_reward(self, row, col, level):
        # start by computing reward
        if self.n_objects == 0 and col == (self.width-1):
            # if no objects are present, goal is to get to the right of plane
            return self.reward_goal
        elif level != 0 and [row, col] == self.object_positions[level - 1]:
            return self.reward_goal
        else:
            return self.reward_default

    def to_id_clamp(self, row, col, level):
        """Convert (x,y) state to state ID, after clamp x & y to lie in grid."""
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        state_id = (self.width * self.height) * level \
                   + self.width * row \
                   + col
        assert 0 <= state_id < self.n_states
        return state_id

    def to_coord(self, state_id):
        """Convert id to (x,y,level)"""
        assert 0 <= state_id < self.n_states
        level = state_id // (self.width * self.height)

        grid_id = state_id % (self.width * self.height)
        col = grid_id % self.width
        row = grid_id // self.width

        return (row, col, level)

    @property
    def observation_matrix(self):
        return self._observation_matrix

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def horizon(self):
        return self._horizon

    @property
    def initial_state_dist(self):
        # always start in s0
        rv = np.zeros((self.n_states,))
        rv[:self.width * self.height] = 1/(self.width * self.height)
        return rv

    def draw_value_vec(self, D) -> None:
        """Use matplotlib to plot a vector of values for each state.

        The vector could represent things like reward, occupancy measure, etc.

        Args:
            D: the vector to plot.
        """
        import matplotlib.pyplot as plt

        grid = D.reshape(self.height, self.width, self.n_levels)

        fig, axs = plt.subplots(self.n_levels, 1)
        for level in range(self.n_levels):
            axs[level, 0].imshow(grid[:, :, level])
            axs[level, 0].gca().grid(False)

def register_grid(suffix, kwargs):
    gym.register(
        f"imitation/GridWorld{suffix}-v0",
        entry_point="imitation.envs.examples.model_envs:GridWorld",
        kwargs=kwargs,
    )


registered_env_names = []

for height, width, horizon in [(4, 7, 9)]:
    for reduce_rows in [0, 1]:
        for reduce_cols in [0, 1]:
            for n_objects in [0, 1, 2, 3]:
                reduce_rows_str = "xred-rows" if reduce_rows else ""
                reduce_cols_str = "xred-cols" if reduce_cols else ""
                objects_str = f"xobjects-{n_objects}"
                env_name = f"{width}x{height}{objects_str}{reduce_rows_str}{reduce_cols_str}"

                registered_env_names.append(env_name)

                register_grid(
                    env_name,
                    kwargs={
                        "width": width,
                        "height": height,
                        "horizon": horizon,
                        "n_objects": n_objects,
                        "reduce_rows": reduce_rows,
                        "reduce_cols": reduce_cols
                    },
                )