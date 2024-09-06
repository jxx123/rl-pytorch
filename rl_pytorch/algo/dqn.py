import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import wandb
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque([], maxlen=size)

    def push(self, state, action, next_state, reward, terminated):
        self.buffer.append((state, action, next_state, reward, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state = torch.as_tensor(
            np.array([step[0] for step in batch]), dtype=torch.float32
        )
        action = torch.as_tensor(
            np.array([step[1] for step in batch]), dtype=torch.int64
        ).view(batch_size, 1)
        next_state = torch.as_tensor(
            np.array([step[2] for step in batch]), dtype=torch.float32
        )
        reward = torch.as_tensor(
            np.array([step[3] for step in batch]), dtype=torch.float32
        ).view(batch_size, 1)

        terminated = torch.as_tensor(
            np.array([1 if step[4] else 0 for step in batch]), dtype=torch.int32
        ).view(batch_size, 1)
        return state, action, next_state, reward, terminated

    def __len__(self):
        return len(self.buffer)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_func()]
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, obs_dim, n_act, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.n_act = n_act
        self.q_net = mlp([obs_dim] + hidden_sizes + [n_act], activation=activation)

    def forward(self, obs):
        """
        Args:
          obs: (obs_dim, )

        Returns:
          q values for each action, the shape is (n_act, )
        """
        return self.q_net(obs)

    def get_action(self, obs: np.ndarray, deterministic=True, epsilon=0.1) -> int:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        q = self.q_net(obs)
        if not deterministic and random.random() < epsilon:
            act = random.randint(0, self.n_act - 1)
            return act

        return torch.argmax(q).numpy().item()


def dqn_train(
    env_name,
    num_eps=1000,
    lr=1e-4,
    hidden_sizes=[32],
    buffer_size=10000,
    render=False,
    batch_size=128,
    seed=0,
    epsilon_start=0.9,
    epsilon_end=0.05,
    epsilon_decay=10000,
    discount=0.98,
    validate_freq=10,
    use_target_network=True,
    target_nework_update_freq=10,
    forgetting_factor=0.95,
):
    env = gym.make(env_name, render_mode="human" if render else None)
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "Only discrete action space is supported."
    buffer = ReplayBuffer(buffer_size)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    qnet = QNet(obs_dim, n_act, hidden_sizes)
    if use_target_network:
        q_target = QNet(obs_dim, n_act, hidden_sizes)
        q_target.load_state_dict(qnet.state_dict())
    # could use Huber Loss as well
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)

    global_step = 0
    for eps in range(num_eps):
        qnet.train()
        obs, _ = env.reset(seed=seed)
        ret = 0
        step = 0
        while True:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * global_step / epsilon_decay
            )
            action = qnet.get_action(obs, deterministic=False, epsilon=epsilon)
            # wandb.log(
            #     {
            #         "q_value": wandb.Histogram(
            #             qnet(obs_random_sample).view(-1).detach().numpy()
            #         )
            #     }
            # )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # print(f"global_step: {global_step}, action: {action}, reward: {reward}")
            ret += reward
            buffer.push(obs, action, next_obs, reward, terminated)

            if len(buffer) >= batch_size:
                (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_terminated,
                ) = buffer.sample(batch_size)

                if use_target_network:
                    with torch.no_grad():
                        next_return_estimate = (
                            discount
                            * q_target(batch_next_obs).max(dim=-1).values.view(-1, 1)
                            * (1 - batch_terminated)
                        )
                else:
                    next_return_estimate = (
                        discount
                        * qnet(batch_next_obs).max(dim=-1).values.view(-1, 1)
                        * (1 - batch_terminated)
                    )
                td_target = batch_rewards + next_return_estimate
                td_pred = qnet(batch_obs).gather(1, batch_actions)
                loss = loss_fn(td_target, td_pred)
                # wandb.log({"loss": loss.item()})
                optimizer.zero_grad()
                loss.backward()
                # wandb.log(
                #     {
                #         f"gradient/{name}": wandb.Histogram(param.grad.cpu().numpy())
                #         for name, param in qnet.named_parameters()
                #     },
                # )
                optimizer.step()

                if use_target_network and global_step % target_nework_update_freq == 0:
                    q_target_state_dict = q_target.state_dict()
                    qnet_state_dict = qnet.state_dict()
                    for name, param in qnet_state_dict.items():
                        q_target_state_dict[name] = (
                            forgetting_factor * q_target_state_dict[name]
                            + (1 - forgetting_factor) * param
                        )
                    q_target.load_state_dict(q_target_state_dict)

            if terminated or truncated:
                wandb.log({"return": ret, "total_step": step + 1})
                print(f"Episode {eps}: total_step: {step + 1}, return: {ret}")
                break

            obs = next_obs
            step += 1
            global_step += 1

        if eps % validate_freq == 0:
            validate(qnet, env, 10, eps)

    obs_random_sample = torch.tensor(
        [env.observation_space.sample() for _ in range(20)]
    )
    q_values = qnet(obs_random_sample)
    print("Q values for random samples: ", q_values)
    return qnet


def validate(qnet, env, num_eps, train_eps):
    qnet.eval()
    returns = []
    for eps in range(num_eps):
        ret = 0
        obs, _ = env.reset(seed=eps + train_eps)
        while True:
            action = qnet.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            if terminated or truncated:
                returns.append(ret)
                break
    mean_return = np.mean(returns)
    print(f"Eps {train_eps}, validation return: {mean_return}")
    wandb.log({"validation_return": mean_return})
    return mean_return


if __name__ == "__main__":
    seed = 10
    num_eps = 2000
    lr = 1e-4
    hidden_sizes = [32]
    buffer_size = 10000
    render = False
    batch_size = 256
    seed = 0
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 5000
    discount = 0.95
    use_target_network = False
    target_nework_update_freq = 1
    forgetting_factor = 0.98

    configs = {
        "seed": seed,
        "num_eps": num_eps,
        "lr": lr,
        "hidden_sizes": hidden_sizes,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "discount": discount,
        "render": render,
        "use_target_network": use_target_network,
        "target_nework_update_freq": target_nework_update_freq,
        "forgetting_factor": forgetting_factor,
    }

    random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    run_name = (
        "DQN with target network" if use_target_network else "DQN no target network"
    )

    wandb.init(project="LunarLander-v2", name=run_name, config=configs)
    qnet = dqn_train(
        "LunarLander-v2",
        seed=seed,
        num_eps=num_eps,
        lr=lr,
        hidden_sizes=hidden_sizes,
        buffer_size=buffer_size,
        render=render,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        discount=discount,
        use_target_network=use_target_network,
        target_nework_update_freq=target_nework_update_freq,
        forgetting_factor=forgetting_factor,
    )
