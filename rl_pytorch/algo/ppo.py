import numpy as np
import wandb
import torch
import gymnasium as gym
import scipy.signal
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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
        self.mu_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation,
            output_activation=nn.Sigmoid,
        )

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
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.GELU
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
        # NOTE: This is a hack for simglucose
        return np.maximum(a.numpy(), [0.0]), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPOBuffer:
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

    def store(self, obs, act, rew, val, logp):
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

    def finish_path(self, last_val=0):
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
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

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
        # print(data)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(
    env_name="CartPole-v1",
    num_epochs=50,
    steps_per_epoch=4000,
    pi_lr=1e-3,
    v_lr=1e-3,
    hidden_sizes=[64, 64],
    gamma=0.99,
    lam=0.97,
    epsilon=0.1,
    train_pi_iters=10,
    train_v_iters=10,
    use_wandb=False,
    use_vpg_loss=False,
    project_name="",
    exp_name="",
    seed=0,
):
    if use_wandb:
        wandb.init(project=project_name, name=exp_name, config=locals())
    seed += 10000
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    ac = MLPActorCritic(
        env.observation_space, env.action_space, hidden_sizes=hidden_sizes
    )

    obs_dim = env.observation_space.shape
    # NOTE: For scalar action the act_dim = (), this is critical, if setting
    # act_dim to 1, the logp computation does some weird broadcast, making the
    # logp shape to (B, B) instead of (B,).
    act_dim = env.action_space.shape

    buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma=gamma, lam=lam)
    pi_opt = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    v_opt = torch.optim.Adam(ac.v.parameters(), lr=v_lr)

    def compute_pi_loss(obs, act, adv, logp_old):
        if use_vpg_loss:
            return compute_vanilla_pi_loss(obs, act, adv, logp_old)
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clamp_ratio = torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon)
        loss = -torch.min((ratio * adv), (clamp_ratio * adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + epsilon) | ratio.lt(1 - epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        return loss, approx_kl, ent, clipfrac

    def compute_vanilla_pi_loss(obs, act, adv, logp_old):
        pi, logp = ac.pi(obs, act)
        loss = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipfrac = 0
        return loss, approx_kl, ent, clipfrac

    def compute_v_loss(obs, ret):
        return ((ac.v(obs) - ret) ** 2).mean()

    def update(ep, data):
        obs, act, adv, ret, logp_old = (
            data["obs"],
            data["act"],
            data["adv"],
            data["ret"],
            data["logp"],
        )

        pi_old_loss, _, _, _ = compute_pi_loss(obs, act, adv, logp_old)
        v_old_loss = compute_v_loss(obs, ret)

        for _ in range(train_pi_iters):
            pi_opt.zero_grad()
            pi_loss, approx_kl, ent, clipfrac = compute_pi_loss(obs, act, adv, logp_old)
            pi_loss.backward()
            pi_opt.step()

        for _ in range(train_v_iters):
            v_opt.zero_grad()
            v_loss = compute_v_loss(obs, ret)
            v_loss.backward()
            v_opt.step()

        delta_pi_loss = pi_old_loss - pi_loss
        delta_v_loss = v_old_loss - v_loss

        if use_wandb:
            wandb.log(
                {
                    "pi_loss": pi_loss,
                    "pi_kl": approx_kl,
                    "clip_frac": clipfrac,
                    "pi_entropy": ent,
                    "v_loss": v_loss,
                    "delta_pi_loss": delta_pi_loss,
                    "delta_v_loss": delta_v_loss,
                },
                step=ep,
            )
        print(
            f"pi_loss: {pi_loss:.3f}, pi_kl: {approx_kl:.3f}, clip_frac: {clipfrac:.3f}, pi_entropy: {ent:.3f}, v_loss: {v_loss:.3f}, delta_pi_loss: {delta_pi_loss:.3f}, delat_v_loss: {delta_v_loss:.3f}"
        )

    obs, _ = env.reset(seed=seed + 20000)
    curr_ret = 0
    curr_eps_len = 0
    for ep in range(num_epochs):
        rets = []
        eps_lens = []

        for t in range(steps_per_epoch):
            # rollout
            act, val, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, rew, terminated, truncated, _ = env.step(act)

            buffer.store(obs, act, rew, val, logp)
            curr_ret += rew
            curr_eps_len += 1
            obs = next_obs

            truncated = truncated or t == steps_per_epoch - 1
            if terminated or truncated:
                if truncated:
                    _, last_val, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    last_val = 0

                buffer.finish_path(last_val=last_val)

                rets.append(curr_ret + last_val)
                if terminated:
                    eps_lens.append(curr_eps_len)

                curr_ret = 0
                curr_eps_len = 0
                obs, _ = env.reset(seed=seed + 50000 + ep + t)

        mean_ret = np.mean(np.array(rets))
        mean_eps_len = np.mean(np.array(eps_lens))
        if use_wandb:
            wandb.log({"return": mean_ret, "eps_len": mean_eps_len}, step=ep)
        print(f"Epoch {ep}: return {mean_ret:.2f}, eps_len {mean_eps_len:.2f}")
        update(ep, buffer.get())


if __name__ == "__main__":
    configs = {
        "env_name": "MountainCarContinuous-v0",
        "hidden_sizes": [256, 256],
        "num_epochs": 100,
        "steps_per_epoch": 4000,
        "train_v_iters": 80,
        "train_pi_iters": 80,
        "gamma": 0.99,
        "lam": 0.97,
        "pi_lr": 1e-3,
        "v_lr": 1e-3,
        "seed": 0,
        "use_wandb": False,
        "project_name": "MountainCarContinuous",
        "exp_name": "ppo",
        "use_vpg_loss": False,
    }

    ppo(**configs)
