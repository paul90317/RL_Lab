import gym
import keyboard
from gym.wrappers import FrameStack
import cv2
import numpy as np
import pickle as pk
import os
def transform(frames):
    new_frames=[]
    for img in frames:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[51:155,8:]
        img = cv2.resize(img,(84,84))
        new_frames.append(img)
    return  np.asarray([new_frames], dtype=np.float32)

# Choose the environment
env = FrameStack(gym.make('ALE/Enduro-v5',render_mode='human'),4)

# Reset the environment to get the initial state
state = env.reset()
action = 0
result = []
if os.path.isfile('pre_train_data.pickle'):
    with open('pre_train_data.pickle','wb') as f:
        result = pk.load(f)

while True:
    env.render()
    if keyboard.is_pressed('5'):
        action = 0
    elif keyboard.is_pressed('8'):
        action = 1
    elif keyboard.is_pressed('6'):
        action = 2
    elif keyboard.is_pressed('4'):
        action = 3
    elif keyboard.is_pressed('2'):
        action = 4
    elif keyboard.is_pressed('3'):
        action = 5
    elif keyboard.is_pressed('1'):
        action = 6
    elif keyboard.is_pressed('9'):
        action = 7
    elif keyboard.is_pressed('7'):
        action = 8
    else:
        action = 0
    obs, reward, terminate, truncate, info = env.step(action)
    transform(obs)
    """
    result.append({
		"observation": transform(obs),
		"action": [action],
		"reward": reward,
		"done": terminate,
	})
    """
    # Break the loop if the episode is done
    if terminate:
        break

# Close the environment
with open('pre_train_data.pickle','wb') as f:
    pk.dump(result, f)
env.close()