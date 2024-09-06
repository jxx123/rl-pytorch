from simglucose.envs import T1DSimGymnaisumEnv
import gymnasium as gym
import numpy as np


class StatefulSimglucoseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        n_bg=12,
        n_insulin=30,
        n_cho=30,
        **kwargs,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimGymnaisumEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            render_mode=render_mode,
            **kwargs,
        )
        self.n_bg = n_bg
        self.n_insulin = n_insulin
        self.n_cho = n_cho

        # Set custom metadata
        for k, v in kwargs.items():
            if k in ["render_modes", "render_fps"]:
                continue
            self.metadata[k] = v

        self.observation_space = gym.spaces.Box(
            low=0,
            high=600,
            shape=(self.n_bg + self.n_insulin + self.n_cho,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )

    def _states(self):
        return np.concatenate(
            (self.bg_hist, self.insulin_hist, self.cho_hist), dtype=np.float32
        )

    @staticmethod
    def _push_buffer(x, new_val):
        x[1:] = x[:-1]
        x[0] = new_val

    def _update_buffers(self, obs, act):
        self._push_buffer(self.bg_hist, obs[0])
        self._push_buffer(self.cho_hist, obs[1])
        # only push the bolus
        self._push_buffer(self.insulin_hist, act[1])

    def step(self, action):
        scaled_action = action[0] + 1.0
        act = np.array([0, scaled_action])
        obs, reward, terminated, truncated, info = self.env.step(act)
        self._update_buffers(obs, act)

        return self._states(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        cgm = obs[0]
        cho = obs[1]
        self.bg_hist = np.ones(self.n_bg) * cgm
        self.insulin_hist = np.zeros(self.n_insulin)
        self.cho_hist = np.concatenate((np.ones(1) * cho, np.zeros(self.n_cho - 1)))
        return self._states(), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
