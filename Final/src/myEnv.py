from racecar_gym.env import RaceEnv
import numpy as np
from settings import action_list, action_space, observation_space, N_frame
from collections import deque

class FinalEnv(RaceEnv):
    austria_competition = 0
    circle_cw_competition_collisionStop = 0
    def __init__(self, testing :bool, i_scenario :int):
        scenarios = ['austria_competition', 'circle_cw_competition_collisionStop']
        super().__init__(scenario=scenarios[i_scenario],
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False)
        self.testing = testing
        self.observation_space = observation_space
        self.lazy_count = 0
        self.action_space = action_space
        self.N_frame = N_frame

    def step(self, actions):
        obs, rew, terminated, truncated, info = super().step(action_list[actions])

        obs = obs[0]
        self.frames.append(obs)

        if info['n_collision'] > 0:
            terminated = True

        if rew == 0:
            self.lazy_count +=1
            if self.lazy_count >= 100:
                truncated = True
        else:
            self.lazy_count = 0
            
        info['original_reward'] = rew
        return np.array(list(self.frames)), 1000 * rew - (terminated or truncated), terminated, truncated, info
        
    def reset(self, seed = 0):
        self.lazy_count = 0
        if self.testing:
            obs, info = super().reset(seed=seed)
        else:
            obs, info = super().reset(options=dict(mode='random'),seed=seed)
        obs = obs[0]
        self.frames = deque([obs for _ in range(self.N_frame)],maxlen = self.N_frame)
        return np.array(list(self.frames)), info
