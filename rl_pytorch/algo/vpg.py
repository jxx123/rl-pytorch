import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from scipy import signal
import wandb
import random


def mlp(sizes, activation=nn.GELU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_func()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


def combined_shape(length, shape):
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class GAEBuffer:
    def __init__(self, obs_dim, act_dim, max_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.adv_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.logp_buf = np.zeros(max_size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.traj_start_idx = 0
        self.max_size = max_size

    def push(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self):
        # reset the pointers
        self.ptr = 0
        self.traj_start_idx = 0

        # Normalize advantage. It is a trick.
        # Jinyu's comment: this normalization does not change the gradient
        # direction but makes the gradient less possible to explode or vanish.
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = {
            "obs": self.obs_buf,
            "act": self.act_buf,
            "ret": self.ret_buf,
            "adv": adv_buf,
            "logp": self.logp_buf,
        }
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def finish_traj(self, last_v=0):
        vals = self.val_buf[self.traj_start_idx : self.ptr]
        vals = np.append(vals, last_v)
        rews = self.rew_buf[self.traj_start_idx : self.ptr]
        rews = np.append(rews, last_v)

        # TD estimates
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[self.traj_start_idx : self.ptr] = discount_cumsum(
            deltas, self.gamma * self.lam
        )
        self.ret_buf[self.traj_start_idx : self.ptr] = discount_cumsum(
            rews, self.gamma
        )[:-1]
        self.traj_start_idx = self.ptr


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def push(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_traj(self, last_v=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_v)
        vals = np.append(self.val_buf[path_slice], last_v)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def count_parameters(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class VanillaPolicyGradient:
    def __init__(
        self,
        env_name: str,
        hidden_sizes: list[int],
        render=False,
        device="cuda",
        use_wandb=True,
    ):
        self.env = gym.make(env_name, render_mode="human" if render else None)
        self.use_wandb = use_wandb
        self.device = torch.device(device)
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, hidden_sizes=hidden_sizes
        )

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        print(f"actor size: {count_parameters(self.ac.pi)}")
        print(f"critic size: {count_parameters(self.ac.v)}")

    def train(
        self,
        num_epoch,
        steps_per_epoch,
        train_v_iters=1,
        gamma=0.99,
        lam=0.95,
        actor_lr=1e-3,
        critic_lr=1e-3,
        seed=0,
    ):
        buffer = VPGBuffer(
            self.obs_dim, self.act_dim, steps_per_epoch, gamma=gamma, lam=lam
        )
        act_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.Adam(self.ac.v.parameters(), lr=critic_lr)

        def compute_actor_loss(data):
            obs, act, adv, logp_old = (
                data["obs"],
                data["act"],
                data["adv"],
                data["logp"],
            )
            pi, logp = self.ac.pi(obs, act)

            approx_kl = (logp_old - logp).mean()
            ent = pi.entropy().mean()
            loss = -(logp * adv).mean()
            return loss, approx_kl, ent

        def compute_loss_pi(data):
            obs, act, adv, logp_old = (
                data["obs"],
                data["act"],
                data["adv"],
                data["logp"],
            )

            # Policy loss
            pi, logp = self.ac.pi(obs, act)
            loss_pi = -(logp * adv).mean()

            # # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()

            return loss_pi, approx_kl, ent

        def compute_critic_loss(data):
            obs, ret = data["obs"], data["ret"]
            v_est = self.ac.v(obs)
            return ((ret - v_est) ** 2).mean()

        def update():
            data = buffer.get()

            act_optimizer.zero_grad()
            # pi_loss, approx_kl, ent = compute_actor_loss(data)
            pi_loss, approx_kl, ent = compute_loss_pi(data)
            pi_loss.backward()
            act_optimizer.step()

            for _ in range(train_v_iters):
                critic_optimizer.zero_grad()
                v_loss = compute_critic_loss(data)
                v_loss.backward()
                critic_optimizer.step()

            if self.use_wandb:
                wandb.log(
                    {
                        "pi_loss": pi_loss,
                        "pi_kl": approx_kl,
                        "pi_entropy": ent,
                        "v_loss": v_loss,
                    }
                )

        for ep in range(num_epoch):
            rets = []
            eps_len = []
            obs, _ = self.env.reset()
            curr_ret = 0
            step_in_eps = 0
            for step in range(steps_per_epoch):
                act, val, logp_a = self.ac.step(
                    torch.as_tensor(obs, dtype=torch.float32)
                )
                next_obs, rew, terminated, truncated, _ = self.env.step(act)

                curr_ret += rew
                step_in_eps += 1
                buffer.push(obs, act, rew, val, logp_a)

                # This is important, don't forget
                obs = next_obs

                truncated = truncated or step == steps_per_epoch - 1
                if terminated or truncated:
                    if truncated:
                        _, last_val, _ = self.ac.step(
                            torch.as_tensor(obs, dtype=torch.float32)
                        )
                        rets.append(curr_ret + last_val)
                    else:
                        last_val = 0
                        eps_len.append(step_in_eps)
                        rets.append(curr_ret)

                    curr_ret = 0
                    step_in_eps = 0
                    buffer.finish_traj(last_v=last_val)
                    obs, _ = self.env.reset()

            mean_eps_len = np.mean(np.array(eps_len))
            mean_ret = np.mean(np.array(rets))
            print(f"Epoch {ep}: return: {mean_ret}, mean_eps_len: {mean_eps_len}")
            if self.use_wandb:
                wandb.log({"return": mean_ret, "mean_eps_len": mean_eps_len})

            update()


if __name__ == "__main__":
    configs = {
        "env_name": "CartPole-v1",
        "hidden_sizes": [64, 64],
        "num_epoch": 100,
        "steps_per_epoch": 4000,
        "train_v_iters": 80,
        "gamma": 0.99,
        "lam": 0.97,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "seed": 0,
        "use_wandb": True,
    }

    if configs["use_wandb"]:
        wandb.init(project="rl-pytorch", name="VPG", config=configs)

    torch.manual_seed(configs["seed"] + 123)
    np.random.seed(configs["seed"] + 456)
    random.seed(configs["seed"] + 1)

    vpg = VanillaPolicyGradient(
        configs["env_name"], configs["hidden_sizes"], use_wandb=configs["use_wandb"]
    )
    vpg.train(
        configs["num_epoch"],
        configs["steps_per_epoch"],
        train_v_iters=configs["train_v_iters"],
        gamma=configs["gamma"],
        lam=configs["lam"],
        actor_lr=configs["actor_lr"],
        critic_lr=configs["critic_lr"],
        seed=configs["seed"],
    )
