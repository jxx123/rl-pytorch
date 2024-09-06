from rl_pytorch.algo.ppo import ppo
from gymnasium.envs.registration import register

register(
    id="simglucose/adult1",
    entry_point="rl_pytorch.envs.simglucose_env:StatefulSimglucoseEnv",
    max_episode_steps=10000,
    kwargs={
        "patient_name": "adult#001",
    },
)

configs = {
    "env_name": "simglucose/adult1",
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
    "use_wandb": True,
    "project_name": "simglucose",
    "exp_name": "ppo",
    "use_vpg_loss": False,
}

ppo(**configs)
