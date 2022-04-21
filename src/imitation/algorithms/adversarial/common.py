"""Core code for adversarial imitation learning, shared between GAIL and AIRL."""
import abc
import collections
import dataclasses
import logging
import os
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, policies, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util
from imitation.scripts.common import common as common_config

import time

def get_wgan_loss_disc(disc_logits, labels_ge_is_one):
    loss = torch.mean(disc_logits[(1 - labels_ge_is_one).bool()]) - torch.mean(disc_logits[labels_ge_is_one.bool()])
    return loss

def get_wgan_loss_gen(disc_logits, labels_ge_is_one):
    loss = torch.mean(disc_logits[labels_ge_is_one.bool()])
    return loss

def compute_train_stats(
    disc_logits_gen_is_high: th.Tensor,
    labels_gen_is_one: th.Tensor,
    disc_loss: th.Tensor,
    prefix="",
    encoder_net=None
) -> Mapping[str, float]:
    """Train statistics for GAIL/AIRL discriminator.

    Args:
        disc_logits_gen_is_high: discriminator logits produced by
            `DiscrimNet.logits_gen_is_high`.
        labels_gen_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        A mapping from statistic names to float values.
    """
    with th.no_grad():
        bin_is_generated_pred = disc_logits_gen_is_high > 0
        bin_is_generated_true = labels_gen_is_one > 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_labels = float(len(labels_gen_is_one))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(logits=disc_logits_gen_is_high)
        entropy = th.mean(label_dist.entropy())

    pairs = [
        (f"disc_loss{prefix}", float(th.mean(disc_loss))),
        # accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        (f"disc_acc{prefix}", float(acc)),
        (f"disc_acc_expert{prefix}", float(expert_acc)),
        (f"disc_acc_gen{prefix}", float(generated_acc)),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        (f"disc_entropy{prefix}", float(entropy)),
        # true number of expert demos and predicted number of expert demos
        (f"disc_proportion_expert_true{prefix}", float(pct_expert)),
        (f"disc_proportion_expert_pred{prefix}", float(pct_expert_pred)),
        (f"n_expert{prefix}", float(n_expert)),
        (f"n_generated{prefix}", float(n_generated)),
    ]  # type: Sequence[Tuple[str, float]]

    if encoder_net is not None and encoder_net.type == "network":
        weights_array = encoder_net.mlp.dense_final.weight.cpu().numpy().flatten()
        for i in range(len(weights_array)):
            pairs = pairs + [(f"zenc_weight_{i}", float(weights_array[i]))]

    return collections.OrderedDict(pairs)


def get_min_max_of_network(network):

    if not hasattr(network, "mlp"):
        return 0, 0

    sequential_network = network.mlp

    min_overall = None
    max_overall = None

    for layer in sequential_network.children():

        if not hasattr(layer, "weight"):
            continue

        min_this_layer = torch.min(layer.weight.data)
        max_this_layer = torch.max(layer.weight.data)
        min_overall = min_this_layer if min_overall is None or min_overall > min_this_layer else min_overall
        max_overall = max_this_layer if max_overall is None or max_overall < max_this_layer else max_overall

    return min_overall, max_overall


def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = (eps * real_data + (1 - eps) * fake_data).detach()
    interpolation.requires_grad = True

    # get logits for interpolated images
    interp_logits = netD.forward_direct(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)

class AdversarialTrainer(base.DemonstrationAlgorithm[types.Transitions]):
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        encoder_net: reward_nets.RewardNet,
        encoder_net_expert: reward_nets.RewardNet,
        n_gen_updates_per_round,
        n_disc_updates_per_round,
        disc_lr,
        n_enc_updates_per_round,
        enc_lr,
        log_dir: str = "output/",
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        use_wgan: bool = False,
        eval_env: None,
    ):
        if disc_opt_cls == "adam":
            disc_opt_cls = th.optim.Adam
        elif disc_opt_cls == "rmsprop":
            disc_opt_cls = th.optim.RMSprop
        else:
            raise NotImplemented("Unknown Discriminator Optimizor class, neither adam nor RMSprop")

        self.demo_batch_size = demo_batch_size
        self._demo_data_loader = None
        self._endless_expert_iterator = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=False,
        )

        self._global_step = 0
        self._disc_step = 0
        self._encoder_step = 0
        self.n_gen_updates_per_round = n_gen_updates_per_round
        self.n_disc_updates_per_round = n_disc_updates_per_round
        self.n_enc_updates_per_round = n_enc_updates_per_round

        self.venv = venv
        self.gen_algo = gen_algo
        self._reward_net = reward_net.to(gen_algo.device)

        self._encoder_net = encoder_net
        self._encoder_net_expert = encoder_net_expert

        if self._encoder_net.type == "network":
            self._encoder_net = self._encoder_net.to(gen_algo.device)

        self._log_dir = log_dir
        self.use_wgan = use_wgan

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph

        self._disc_opt = self._disc_opt_cls(
            self._reward_net.parameters(),
            lr=disc_lr
        )

        if self._encoder_net.type == "network":
            self._encoder_opt = self._disc_opt_cls(
                self._encoder_net.parameters(),
                lr=enc_lr,
            )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
            venv,
            self.reward_train.predict,
            encoder=self._encoder_net
        )
        self.venv_eval_wrapped = reward_wrapper.RewardVecEnvWrapper(
            eval_env,
            self.reward_train.predict,
            encoder=self._encoder_net
        )

        self.gen_callback = self.venv_wrapped.make_log_callback()

        self.venv_train = self.venv_wrapped
        self.venv_eval = self.venv_eval_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

    @property
    def policy(self) -> policies.BasePolicy:
        return self.gen_algo.policy

    @abc.abstractmethod
    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.

        Args:
            state: state at time t, of shape `(batch_size,) + statse_shape`.
            action: action taken at time t, of shape `(batch_size,) + action_shape`.
            next_state: state at time t+1, of shape `(batch_size,) + state_shape`.
            done: binary episode completion flag after action at time t,
                of shape `(batch_size,)`.
            log_policy_act_prob: log probability of generator policy taking
                `action` at time t.

        Returns:
            Discriminator logits of shape `(batch_size,)`. A high output indicates a
            generator-like transition.
        """  # noqa: DAR202

    @property
    @abc.abstractmethod
    def reward_train(self) -> reward_nets.RewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> reward_nets.RewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)


    def train_disc(
        self,
    ) -> Optional[Mapping[str, float]]:

        # compute loss
        batch = self._make_disc_train_batch()

        disc_logits = self.logits_gen_is_high(
            batch["state"],
            batch["action"],
            batch["next_state"],
            batch["done"],
            batch["log_policy_act_prob"],
        )

        if self.use_wgan:
            loss = get_wgan_loss_disc(disc_logits=disc_logits, labels_ge_is_one=batch["labels_gen_is_one"])
        else:
            loss = F.binary_cross_entropy_with_logits(
                disc_logits,
                batch["labels_gen_is_one"].float(),
            )

        # do gradient step
        self._disc_opt.zero_grad()

        state_x = batch["state"]
        next_state_x = batch["next_state"]
        labels_x = batch["labels_gen_is_one"]

        with networks.evaluating(self._reward_net):
            real_data = torch.cat((state_x[~labels_x.bool()], next_state_x[~labels_x.bool()]), dim=1)
            fake_data = torch.cat((state_x[labels_x.bool()], next_state_x[labels_x.bool()]), dim=1)
            gradient_penalty = compute_gp(self._reward_net, real_data, fake_data)

        gp_weight = 10
        loss = loss + gp_weight * gradient_penalty

        loss.backward()
        self._disc_opt.step()
        self._disc_step += 1

        # compute/write stats and TensorBoard data
        if self._disc_step % 100 == 0:
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_gen_is_one"],
                    loss
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)

        return None

    def train_enc(
        self,
    ) -> Optional[Mapping[str, float]]:

        # compute loss
        batch = self._make_enc_train_batch()

        disc_logits = self.logits_gen_is_high_for_encoder(
            batch["state"],
            batch["action"],
            batch["next_state"],
            batch["done"],
            batch["log_policy_act_prob"],
        )

        ## we want the discriminator to predict that this is actually the expert
        labels = batch["labels_gen_is_one"]*0

        if self.use_wgan:
            loss = get_wgan_loss_gen(disc_logits, batch["labels_gen_is_one"])
        else:
            loss = F.binary_cross_entropy_with_logits(
                disc_logits,
                labels.float()  ## TODO make sure this is correct,
            )

        # do gradient step
        self._encoder_opt.zero_grad()
        loss.backward()
        self._encoder_opt.step()
        self._encoder_step += 1

        if self._encoder_step % 100 == 0:
            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_gen_is_one"],
                    loss,
                    prefix="_enc",
                    encoder_net=self._encoder_net
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._encoder_step)

        return None

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:

        raise NotImplemented

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:


        n_rounds = total_timesteps // 1000

        eval_freq = 800

        total_timesteps_in = total_timesteps

        # eval_env = self.venv_eval
        # n_eval_episodes = 3

        total_timesteps, callback_gen = self.gen_algo._setup_learn(
            total_timesteps=total_timesteps_in,
            callback=None,
            eval_env=None,
            # eval_env=eval_env
            # eval_freq=eval_freq,
            # n_eval_episodes=n_eval_episodes,
            reset_num_timesteps=False,
            tb_log_name="run",
        )

        scheduler_actor_lr = th.optim.lr_scheduler.ExponentialLR(self.gen_algo.actor.optimizer, gamma=0.5**(1/100000)) # TODO verify that this actually works

        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):

            start_time = time.time()

            callback_gen.on_training_start(locals(), globals())

            action_noise = "random" if self.gen_algo.num_timesteps < common_config.get_dac_parameters()["random_actions"] else self.gen_algo.action_noise

            rollout_gen = self.gen_algo.collect_rollouts(
                self.gen_algo.env,
                train_freq=self.gen_algo.train_freq,
                action_noise=action_noise,
                callback=callback_gen,
                learning_starts=self.gen_algo.learning_starts,
                replay_buffer=self.gen_algo.replay_buffer,
                log_interval=4,
            )

            if self.gen_algo.num_timesteps > self.gen_algo.learning_starts:

                with self.logger.accumulate_means("disc"):
                    for _ in range(self.n_disc_updates_per_round):
                        with networks.training(self.reward_train):
                            self.train_disc()

                with self.logger.accumulate_means("enc"):
                    if self._encoder_net.type == "network":
                        for _ in range(self.n_enc_updates_per_round):
                                self.train_enc()

                with self.logger.accumulate_means("gen"):
                    update_actor = True if self.gen_algo.num_timesteps > self.gen_algo.learning_starts + common_config.get_dac_parameters()["policy_updates_delay"] else False
                    self.gen_algo.train(batch_size=self.gen_algo.batch_size, gradient_steps=self.n_gen_updates_per_round, update_actor=update_actor)
                    scheduler_actor_lr.step()

            callback_gen.on_training_end()

            self._global_step += 1

            self.logger.record("actor_lr", self.gen_algo.actor.optimizer.param_groups[0]["lr"])
            self.logger.record("critic_lr", self.gen_algo.critic.optimizer.param_groups[0]["lr"])
            self.logger.record("disc_lr", self._disc_opt.param_groups[0]["lr"])
            self.logger.record("enc_lr", self._encoder_opt.param_groups[0]["lr"]) if self._encoder_net.type == "network" else None
            self.logger.record("train_it_time", time.time()-start_time)

            if callback:
                callback(r)
            self.logger.dump(self._global_step)

            # if r % 10 == 0:
            #     for network, iden in zip([self._reward_net, self._encoder_net], ["disc", "enc"]):
            #         minimum, maximum = get_min_max_of_network(network)
            #         self.logger.record(f"{iden}/min", minimum)
            #         self.logger.record(f"{iden}/max", maximum)

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.reward_train.device)

    def _make_disc_train_batch(
        self,
    ) -> Mapping[str, th.Tensor]:

        expert_samples = self._next_expert_batch()
        gen_samples_inter = self.gen_algo.replay_buffer.sample(self.demo_batch_size)
        gen_samples = {
            "obs": gen_samples_inter.observations,
            "acts": gen_samples_inter.actions,
            "next_obs": gen_samples_inter.next_observations,
            "dones": gen_samples_inter.dones.squeeze(),
        }

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.demo_batch_size):
            raise ValueError("Unknown Error")

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)],
        )

        gen_obs = gen_samples["obs"].detach().to(th.float32)
        gen_next_obs = gen_samples["next_obs"].detach().to(th.float32)
        gen_dones = gen_samples["dones"].detach().to(th.float32)

        expert_obs = expert_samples["obs"].detach().to(th.float32)
        expert_next_obs = expert_samples["next_obs"].detach().to(th.float32)
        expert_dones = expert_samples["dones"].detach().to(th.float32)

        if self._reward_net.device != expert_samples["obs"].device:
            expert_obs = expert_obs.to(self._reward_net.device)  # TODO-tim: put entire demo batch on CPU/CUDA already at beginning
            expert_next_obs = expert_next_obs.to(self._reward_net.device)
            expert_dones = expert_dones.to(self._reward_net.device)

        with th.no_grad():
            gen_obs = self._encoder_net.forward(gen_obs)
            gen_next_obs = self._encoder_net.forward(gen_next_obs)

            expert_obs = self._encoder_net_expert.forward(expert_obs)
            expert_next_obs = self._encoder_net_expert.forward(expert_next_obs)

        obs = th.cat((expert_obs, gen_obs))
        next_obs = th.cat((expert_next_obs, gen_next_obs))
        dones = th.cat((expert_dones, gen_dones))

        batch_dict = {
            "state": obs,
            "action": None,
            "next_state": next_obs,
            "done": dones,
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
            "log_policy_act_prob": None,
        }

        return batch_dict


    def _make_enc_train_batch(
        self,
    ) -> Mapping[str, th.Tensor]:
        """Build and return training batch for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Returns:
            The training batch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """

        gen_samples_inter = self.gen_algo.replay_buffer.sample(int(self.demo_batch_size/2)) # TODO-tim verify it makes sense to train encoder with half the samples
        gen_samples = {
            "obs": gen_samples_inter.observations,
            "acts": gen_samples_inter.actions,
            "next_obs": gen_samples_inter.next_observations,
            "dones": gen_samples_inter.dones.squeeze(),
        }

        n_gen = len(gen_samples["obs"])

        # Guarantee that Mapping arguments are in mutable form.
        gen_samples = dict(gen_samples)

        labels_gen_is_one = np.ones(n_gen, dtype=int)
        labels_is_one_th = self._torchify_array(labels_gen_is_one)

        gen_obs = gen_samples["obs"].detach().to(th.float32)
        gen_next_obs = gen_samples["next_obs"].detach().to(th.float32)
        gen_dones = gen_samples["dones"].detach().to(th.float32)

        gen_obs = self._encoder_net.forward(gen_obs)
        gen_next_obs = self._encoder_net.forward(gen_next_obs)

        batch_dict = {
            "state": gen_obs,
            "action": None,
            "next_state": gen_next_obs,
            "done": gen_dones,
            "labels_gen_is_one": labels_is_one_th,
            "log_policy_act_prob": None,
        }

        return batch_dict
