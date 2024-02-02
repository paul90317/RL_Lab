from gymnasium.spaces import Box, Discrete
import numpy as np
action_list = [(1,-1),(1,1),
              (-0.5,1),(-0.5,-1)]
N_frame = 8
observation_space = Box(0,255,(N_frame,128,128),dtype=np.uint8)
action_space = Discrete(len(action_list))