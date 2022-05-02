"""Constructs deep network reward models."""

import abc
from typing import Callable, Iterable, Sequence, Tuple

import gym
import numpy as np
import torch as th
import torch.nn
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.util import networks


class RewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        input_dimension,
    ):
        """Initialize the RewardNet.
        """
        self.input_dimension = input_dimension
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_th = th.as_tensor(state, device=self.device).to(th.float32)
        action_th = th.as_tensor(action, device=self.device).to(th.float32) if action is not None else None
        next_state_th = th.as_tensor(next_state, device=self.device).to(th.float32)
        done_th = th.as_tensor(done, device=self.device).to(th.float32)

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        if action_th is not None:
            assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        encoder_net
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Preprocesses the inputs, converting between Torch
        tensors and NumPy arrays as necessary.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,`).
        """

        with networks.evaluating(self):
            # switch to eval mode (affecting normalization, dropout, etc)

            state_th, action_th, next_state_th, done_th = self.preprocess(
                state,
                action,
                next_state,
                done,
            )

            with th.no_grad():
                if encoder_net is not None:
                    state_th = encoder_net.forward(state_th)
                    next_state_th = encoder_net.forward(next_state_th)

                rew_th = self(state_th, action_th, next_state_th, done_th)

            rew = rew_th.detach().cpu().numpy().flatten()
            assert rew.shape == state.shape[:1]
            return rew

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()

class BasicRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        input_dimension: int,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(input_dimension)

        full_build_mlp_kwargs = {
            "hid_sizes": (256, 256),
        }
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size": input_dimension,
                "out_size": 1,
                "squeeze_output": True
            },
        )

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)


    def forward(self, state, action, next_state, done):

        inputs = []

        inputs.append(th.flatten(state, 1))
        inputs.append(th.flatten(next_state, 1))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)

        assert outputs.shape == state.shape[:1]

        return outputs

    def forward_direct(self, input):

        outputs = self.mlp(input)

        return outputs

class EncoderNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        target_states,
        normalize_images: bool = True,

    ):
        """Initialize the RewardNet.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images
        self.target_states = target_states

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """

        if self.target_states is not None:
            state = state[:, self.target_states]
            next_state = next_state[:, self.target_states]

        state_th = th.as_tensor(state, device=self.device)
        action_th = th.as_tensor(action, device=self.device) if action is not None else None
        next_state_th = th.as_tensor(next_state, device=self.device)
        done_th = th.as_tensor(done, device=self.device)

        del state, action, next_state, done  # unused

        # preprocess
        state_th = preprocessing.preprocess_obs(
            state_th,
            self.observation_space,
            self.normalize_images,
        )
        action_th = preprocessing.preprocess_obs(
            action_th,
            self.action_space,
            self.normalize_images,
        ) if action_th is not None else None

        next_state_th = preprocessing.preprocess_obs(
            next_state_th,
            self.observation_space,
            self.normalize_images,
        )
        done_th = done_th.to(th.float32)

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        if action_th is not None:
            assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th

    def predict(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Preprocesses the inputs, converting between Torch
        tensors and NumPy arrays as necessary.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,`).
        """

        raise NotImplemented

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()

class BasicEncoderNet(EncoderNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        output_dim: int,
        target_states,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         target_states=target_states)

        self.output_dim = output_dim

        if target_states is None:
            self.input_dim = preprocessing.get_flattened_obs_dim(observation_space)
        else:
            self.input_dim = len(target_states)

        self.type = "network"

        full_build_mlp_kwargs = {
            "hid_sizes": [],
        }
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size": self.input_dim,
                "out_size": self.output_dim,
                "squeeze_output": False,
                "flatten_input": False,
                "normalize_input_layer": False,
                "use_bias": False
            },
        )

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)


    def forward(self, input):
        if self.target_states is not None:
            input = input[:, self.target_states]

        attention = self.mlp(input)
        attention = torch.reshape(attention, (-1, input.shape[1], input.shape[1]))
        outputs = torch.bmm(attention, input.unsqueeze(-1))
        outputs = outputs.squeeze(-1)
        return outputs

class BasicEncoder:

    def __init__(
        self,
        output_dimension,
        target_states
    ):
        self.output_dimension = output_dimension
        self.target_states = target_states
        self.type = "static"

    def forward(self, input):
        if self.target_states is not None:
            input = input[:, self.target_states]
        return input
