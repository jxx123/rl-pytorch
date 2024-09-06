import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import wandb


def compute_return_to_go(rewards):
    total_return = sum(rewards)
    rtg = [total_return - sum(rewards[:i]) for i in range(len(rewards))]
    return rtg


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_func()]
    return nn.Sequential(*layers)


class RTGPolicyGradient:
    def __init__(
        self,
        env_name: str,
        hidden_sizes: list[int],
        render=False,
        device="cuda",
        use_wandb=True,
    ):
        """Return-to-go Policy Gradient that use the full return as the weights."""
        self.env = gym.make(env_name, render_mode="human" if render else None)
        self.use_wandb = use_wandb
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

    def train_epoch(self, epoch, optimizer, num_eps, seed=0, use_rtg=True):
        # s0, s1, ..., sT1; s0, s1, ..., sT2
        batch_obs = []
        # a0, a1, ..., aT1; a0, a1, ..., aT2
        batch_act = []
        # R1, R1, ..., R1; R2, R2, ..., R2
        batch_rtg = []

        steps_count = []
        returns = []
        for _ in range(num_eps):
            rewards = []
            obs, _ = self.env.reset(seed=seed)
            step = 0
            while True:
                act = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(act)
                rewards.append(reward)
                batch_obs.append(obs)
                batch_act.append(act)

                if truncated or terminated:
                    if use_rtg:
                        batch_rtg += compute_return_to_go(rewards)
                    else:
                        batch_rtg += [sum(rewards)] * (step + 1)
                    break

                obs = next_obs
                step += 1

            steps_count.append(step)
            returns.append(sum(rewards))

        loss = self.compute_loss(
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(self.device),
            torch.as_tensor(np.array(batch_act), dtype=torch.int32).to(self.device),
            torch.as_tensor(np.array(batch_rtg), dtype=torch.float32).to(self.device),
        )
        optimizer.zero_grad()
        loss.backward()
        if self.use_wandb:
            wandb.log(
                {
                    f"gradient/{name}": wandb.Histogram(param.grad.cpu().numpy())
                    for name, param in self.actor_net.named_parameters()
                },
                step=epoch,
            )
        optimizer.step()
        return np.mean(steps_count), np.mean(returns), loss

    def train(self, num_epochs, num_eps, lr=1e-2, seed=0, use_rtg=True):
        optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr)
        for i in range(num_epochs):
            avg_eps_len, avg_return, loss = self.train_epoch(
                i, optimizer, num_eps, seed=seed, use_rtg=use_rtg
            )
            if self.use_wandb:
                wandb.log(
                    {"eps_len": avg_eps_len, "return": avg_return, "loss": loss}, step=i
                )
            print(
                f"Epoch: {i}, eps_len: {avg_eps_len:.2f}, return: {avg_return:.2f}, loss: {loss:.2f}"
            )


if __name__ == "__main__":
    env_name = "CartPole-v1"
    seed = 0
    num_epochs = 100
    num_eps = 30
    hidden_sizes = [32]
    render = False
    device = "cpu"
    lr = 1e-2
    use_rtg = False
    configs = {
        "seed": seed,
        "num_epochs": num_epochs,
        "num_eps": num_eps,
        "hidden_sizes": hidden_sizes,
        "render": render,
        "device": device,
        "lr": lr,
        "use_rtg": use_rtg,
    }

    wandb.init(
        project="RTGPolicyGradient",
        name="Use RTG" if use_rtg else "Use Full Return",
        config=configs,
    )
    torch.manual_seed(seed + 1)
    spg = RTGPolicyGradient(
        env_name, hidden_sizes=hidden_sizes, render=render, device=device
    )
    spg.train(num_epochs, num_eps, lr=lr, seed=seed, use_rtg=use_rtg)
