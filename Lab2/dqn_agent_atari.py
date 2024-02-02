import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
from gym.wrappers import FrameStack
import cv2
from replay_buffer.replay_buffer import ReplayMemory
import sys

def transform(frames):
	new_frames=[]
	for img in frames:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = img[:172,:]
		img = cv2.resize(img,(84,84))
		new_frames.append(img)
	return np.array(new_frames)

class MyReplayMemory(ReplayMemory):
	def __init__(self, capacity, action_space_n):
		super().__init__(capacity)
		self.action_space_n = action_space_n
	def append(self, *transition):
		state, action, reward, next_state, done = transition

		state=transform(state)
		next_state=transform(next_state)
		
		# cv2.imwrite("1.png",state[-1])

		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size, device):
		transitions = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = zip(*transitions)
		return (
			torch.tensor(np.array(state),dtype=torch.float,device=device),
			torch.tensor(action,dtype=torch.int64,device=device),
			torch.tensor(reward,dtype=torch.float,device=device),
			torch.tensor(np.array(next_state),dtype=torch.float,device=device),
			1 - torch.tensor(done,dtype=torch.float,device=device)
		)

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)

		### TODO ###
		# initialize env
		# self.env = ???
		
		self.test_env = FrameStack(gym.make(config['env_id'],render_mode='human'),4)
		self.env = FrameStack(gym.make(config['env_id']),4)

		self.replay_buffer = MyReplayMemory(int(config["replay_buffer_capacity"]),self.env.action_space.n)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)

		if len(sys.argv) > 1:
			self.load(sys.argv[1])
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space : gym.Space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection

		if random.random() < epsilon:
			return random.randint(0, action_space.n-1)
		
		with torch.no_grad():
			x=torch.tensor(np.array([transform(observation)]),dtype=torch.float, device=self.device)
			y=self.behavior_net(x)
			return int(torch.argmax(y))

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, yet = self.replay_buffer.sample(self.batch_size, self.device)
		self.behavior_net.train()

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get max_a Q(s',a) from target net
		# 2. calculate Q_target = r + gamma * max_a Q(s',a)
		# 3. get Q(s,a) from behavior net
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		with torch.no_grad():
			q_max : torch.Tensor = self.behavior_net(next_state)
			q_max = torch.argmax(q_max, dim=1).reshape(self.batch_size,1)
			q_next = self.target_net(next_state)
			q_next : torch.Tensor = q_next.gather(1,q_max)
		
			# if episode terminates at next_state, then q_target = reward
			q_target = self.gamma * q_next * yet + reward
        
		q_value : torch.Tensor = self.behavior_net(state)
		q_value = q_value.gather(1,action)

		criterion = torch.nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		self.behavior_net.eval()
	
	