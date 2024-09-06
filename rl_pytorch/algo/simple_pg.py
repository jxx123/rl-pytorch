import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import wandb

wandb.init(project="SimplePolicyGradient")


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_func()]
    return nn.Sequential(*layers)


class SimplePolicyGradient:
    def __init__(
        self, env_name: str, hidden_sizes: list[int], render=False, device="cuda"
    ):
        """Simple Policy Gradient that use the full return as the weights."""
        self.env = gym.make(env_name, render_mode="human" if render else None)
        self.device = torch.device(device)
        assert isinstance(
            self.env.observation_space, gym.spaces.Box
        ), "Only box observation space is supported."
        assert isinstance(
            self.env.action_space, gym.spaces.Discrete
        ), "Only discrete action space is supported."
        obs_dim = self.env.observation_space.shape[0]
        n_act = self.env.action_space.n
        self.actor_net = mlp([obs_dim] + hidden_sizes + [n_act]).to(device)
        wandb.watch(self.actor_net)

    def get_policy(self, state):
        logits = self.actor_net(state)
        return Categorical(logits=logits)

    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        return self.get_policy(state).sample().item()

    def compute_loss(self, states, actions, weights):
        """Compute the surrogate loss whose gradient happens to be the policy gradient"""
        # We want to maximize the expected return, but pytorch only does
        # gradient descent, hence negative.
        logp = self.get_policy(states).log_prob(actions)
        return -(logp * weights).mean()

    def train_epoch(self, epoch, optimizer, num_eps, seed=0):
        # s0, s1, ..., sT1; s0, s1, ..., sT2
        batch_obs = []
        # a0, a1, ..., aT1; a0, a1, ..., aT2
        batch_act = []
        # R1, R1, ..., R1; R2, R2, ..., R2
        batch_ret = []

        steps_count = []
        returns = []
        for _ in range(num_eps):
            ret = 0
            obs, _ = self.env.reset(seed=seed)
            step = 0
            while True:
                act = self.get_action(obs)
                next_obs, reward, truncated, terminated, info = self.env.step(act)
                ret += reward
                batch_obs.append(obs)
                batch_act.append(act)

                if truncated or terminated:
                    batch_ret += [ret] * (step + 1)
                    break

                obs = next_obs
                step += 1

            steps_count.append(step)
            returns.append(ret)

        loss = self.compute_loss(
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(self.device),
            torch.as_tensor(np.array(batch_act), dtype=torch.int32).to(self.device),
            torch.as_tensor(np.array(batch_ret), dtype=torch.float32).to(self.device),
        )
        optimizer.zero_grad()
        loss.backward()
        wandb.log(
            {
                f"gradient/{name}": wandb.Histogram(param.grad.cpu().numpy())
                for name, param in self.actor_net.named_parameters()
            },
            step=epoch,
        )
        optimizer.step()
        return np.mean(steps_count), np.mean(returns), loss

    def train(self, num_epochs, num_eps, lr=1e-2, seed=0):
        optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        for i in range(num_epochs):
            avg_eps_len, avg_return, loss = self.train_epoch(
                i, optimizer, num_eps, seed=seed
            )
            wandb.log(
                {"eps_len": avg_eps_len, "return": avg_return, "loss": loss}, step=i
            )
            print(
                f"Epoch: {i}, eps_len: {avg_eps_len}, return: {avg_return}, loss: {loss}"
            )


if __name__ == "__main__":
    torch.manual_seed(1)
    spg = SimplePolicyGradient(
        "CartPole-v1", hidden_sizes=[32], render=False, device="cpu"
    )
    spg.train(100, 30, lr=1e-2, seed=0)
