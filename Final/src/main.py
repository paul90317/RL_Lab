from stable_baselines3 import PPO
from myEnv import FinalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from settings import N_frame

def env_fn(testing=False):
    def f():
        return FinalEnv(testing, FinalEnv.austria_competition)
    return f


if __name__ == '__main__':
    envs = SubprocVecEnv([env_fn()] * 4)
    test_env = FinalEnv(True, FinalEnv.austria_competition)
    torch.multiprocessing.freeze_support()

    model = PPO("CnnPolicy", envs, verbose=1, batch_size=512, n_steps=4096)
    learn_per_time = 10000
    for i in range(100):
        model.learn(total_timesteps=learn_per_time)
        
        obs, info = test_env.reset()

        done = False
        total_rew = 0
        local_step = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rew, term, trun, info = test_env.step(action)

            done = term or trun
            
            total_rew += info['original_reward']
            local_step += 1

        model.save(f'models/model_{(i+1)}_{int(total_rew * 100)}_{local_step}.pth')

    envs.close()
    test_env.close()