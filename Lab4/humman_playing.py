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
env = FrameStack(gym.make('CarRacing-v2'),4)

# Reset the environment to get the initial state
state = env.reset()
for i in range(10):
    obs, reward, terminate, truncate, info = env.step([0,0,0])
#cv2.imshow('temp',transform(obs))
print(obs[0].shape)
cv2.imwrite('temp.png',obs[0])
input()
