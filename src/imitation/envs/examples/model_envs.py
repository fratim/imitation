"""Example discrete MDPs for use with tabular MCE IRL."""

from typing import Optional

import gym
import numpy as np

from imitation.envs.resettable_env import TabularModelEnv


class CliffWorld(TabularModelEnv):
    """A grid world with a goal next to a cliff the agent may fall into.

    Illustration::

         0 1 2 3 4 5 6 7 8 9
        +-+-+-+-+-+-+-+-+-+-+  Wind:
      0 |S|C|C|C|C|C|C|C|C|G|
        +-+-+-+-+-+-+-+-+-+-+  ^ ^ ^
      1 | | | | | | | | | | |  | | |
        +-+-+-+-+-+-+-+-+-+-+
      2 | | | | | | | | | | |  ^ ^ ^
        +-+-+-+-+-+-+-+-+-+-+  | | |

    Aim is to get from S to G. The G square has reward +10, the C squares
    ("cliff") have reward -10, and all other squares have reward -1. Agent can
    move in all directions (except through walls), but there is 30% chance that
    they will be blown upwards by one more unit than intended due to wind.
    Optimal policy is to go out a bit and avoid the cliff, but still hit goal
    eventually.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        horizon: int,
        use_xy_obs: bool,
        rew_default: int = -1,
        rew_goal: int = 10,
        rew_cliff: int = -10,
        fail_p: float = 0,
    ):
        """Builds CliffWorld with specified dimensions and reward."""
        super().__init__()
        # assert (
        #     width >= 3 and height >= 2
        # ), "degenerate grid world requested; is this a bug?"
        self.width = width
        self.height = height

        self.object_positions = [[1, 2], [3, 3]]
        self.n_objects = len(self.object_positions)

        succ_p = 1 - fail_p

        n_states = width * height * (self.n_objects + 1)
        O_mat = self._observation_matrix = np.zeros(
            (n_states, 2 if use_xy_obs else n_states),
            dtype=np.float32,
        )

        n_actions = 4 + self.n_objects + 1

        R_vec = self._reward_matrix = np.zeros((n_states,))
        T_mat = self._transition_matrix = np.zeros((n_states, n_actions, n_states))
        self._horizon = horizon

        def to_id_clamp(row, col, obj_level):
            """Convert (x,y) state to state ID, after clamp x & y to lie in grid."""
            row = min(max(row, 0), height - 1)
            col = min(max(col, 0), width - 1)
            state_id = obj_level * (width * height) + row * width + col
            assert 0 <= state_id < self.n_states
            return state_id

        for row in range(height):
            for col in range(width):
                for obj in range(self.n_objects + 1):
                    state_id = to_id_clamp(row, col, obj)

                    # start by computing reward
                    if obj != 0 and [row, col] == self.object_positions[obj-1]:
                        R_vec[state_id] = 10
                    elif obj != 0 and (self.height == self.width == 1):
                        R_vec[state_id] = 10
                    else:
                        R_vec[state_id] = -1

                    # now compute observation
                    if use_xy_obs:
                        raise NotImplementedError
                        # (x, y) coordinate scaled to (0,1)
                        # O_mat[state_id, :] = [
                        #     float(col) / (width - 1),
                        #     float(row) / (height - 1),
                        #     float(obj) / (self.objects + 1 - 1),
                        # ]
                    else:
                        # our observation matrix is just the identity; observation
                        # is an indicator vector telling us exactly what state
                        # we're in
                        O_mat[state_id, state_id] = 1

                    # finally, compute transition matrix entries for each of the
                    # four actions

                    for drow in [-1, 1]:
                        for dcol in [-1, 1]:
                            action_id = (drow + 1) + (dcol + 1) // 2
                            if obj == 0:
                                target_state = to_id_clamp(row + drow, col + dcol, obj)
                                T_mat[state_id, action_id, target_state] += 1
                            else:
                                T_mat[state_id, action_id, state_id] += 1

                    for toggle in range(self.n_objects + 1):
                        action_id = 4 + toggle
                        target_state = to_id_clamp(row, col, toggle)
                        if toggle == 0: # toggling zero always leads back to zero level
                            T_mat[state_id, action_id, target_state] += 1
                        elif obj == 0 and ([row, col] == self.object_positions[toggle - 1]): # correct position to toggle object
                            T_mat[state_id, action_id, target_state] += 1
                        elif obj == 0 and (self.width == self.height == 1): # if environment is reduced, toggle must be possible in (0,0) coordinate
                            T_mat[state_id, action_id, target_state] += 1
                        else: # not in correct position to toggle object so nothing happens
                            T_mat[state_id, action_id, state_id] += 1

        assert np.allclose(np.sum(T_mat, axis=-1), 1, rtol=1e-5), (
            "un-normalised matrix %s" % O_mat
        )

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

        grid = D.reshape(self.height, self.width)
        plt.imshow(grid)
        plt.gca().grid(False)

class GridWorld(CliffWorld):
    def __init__(
            self,
            *,
            width: int,
            height: int,
            horizon: int,
            use_xy_obs: bool,
            rew_default: int = -1,
            rew_goal: int = 10,
            rew_cliff: int = -10,
            fail_p: float = 0.0,
    ):
        super().__init__(
            width=width,
            height=height,
            horizon=horizon,
            use_xy_obs=use_xy_obs,
            rew_default=rew_default,
            rew_goal=rew_goal,
            rew_cliff=rew_cliff,
            fail_p=fail_p
            )

        # def to_id_clamp(row, col):
        #     """Convert (x,y) state to state ID, after clamp x & y to lie in grid."""
        #     row = min(max(row, 0), height - 1)
        #     col = min(max(col, 0), width - 1)
        #     state_id = row * width + col
        #     assert 0 <= state_id < self.n_states
        #     return state_id
        #
        # for row in range(self.height):
        #     for col in range(self.width):
        #         state_id = to_id_clamp(row, col)
        #
        #         # start by computing reward
        #         if col == width - 1:
        #             r = rew_goal  # goal
        #         else:
        #             r = rew_default  # blank
        #
        #         self._reward_matrix[state_id] = r

def register_cliff(suffix, kwargs):
    gym.register(
        f"imitation/CliffWorld{suffix}-v0",
        entry_point="imitation.envs.examples.model_envs:CliffWorld",
        kwargs=kwargs,
    )

def register_grid(suffix, kwargs):
    gym.register(
        f"imitation/GridWorld{suffix}-v0",
        entry_point="imitation.envs.examples.model_envs:GridWorld",
        kwargs=kwargs,
    )


for width, height, horizon in [(1, 1, 9), (7, 1, 9), (7, 4, 9), (15, 6, 18), (100, 20, 110)]:
    for use_xy in [False, True]:
        use_xy_str = "XY" if use_xy else ""
        register_cliff(
            f"{width}x{height}{use_xy_str}",
            kwargs={
                "width": width,
                "height": height,
                "use_xy_obs": use_xy,
                "horizon": horizon,
            },
        )

        register_grid(
            f"{width}x{height}{use_xy_str}",
            kwargs={
                "width": width,
                "height": height,
                "use_xy_obs": use_xy,
                "horizon": horizon,
            },
        )

# These parameter choices are somewhat arbitrary.
# We anticipate most users will want to construct RandomMDP directly.
gym.register(
    "imitation/Random-v0",
    entry_point="imitation.envs.examples.model_envs:RandomMDP",
    kwargs={
        "n_states": 16,
        "n_actions": 3,
        "branch_factor": 2,
        "horizon": 20,
        "random_obs": True,
        "obs_dim": 5,
        "generator_seed": 42,
    },
)