import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise
import sys

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)

		# load weight
		if len(sys.argv) == 3:
			print(sys.argv[1])
			self.load(sys.argv[1])
			self.total_time_step = int(sys.argv[2])
		elif len(sys.argv) != 1:
			exit('unvalid arguments')
	
	def s_epsilon(self, sigma):
		c = 2 * sigma
		noise = self.noise.generate()
		noise = torch.tensor(noise, dtype = torch.float, device = self.device)
		return torch.clip(sigma * noise, -c, c)
	
	def m_epsilon(self, sigma):
		return torch.stack([self.s_epsilon(sigma) for _ in range(self.batch_size)])
	
	def action_clip(self, action):
		return torch.clip(action, torch.tensor([-1, 0, 0], dtype = torch.float, device = self.device), torch.tensor(1, dtype = torch.float, device = self.device))

	def decide_agent_actions(self, state, sigma = 0.0):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		with torch.no_grad():
			state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
			noise = self.s_epsilon(sigma)
			action = self.actor_net(state)[0] + noise
			action = self.action_clip(action)

		# return action
		return action.cpu().numpy()
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
			# select action a_next from target actor network and add noise for smoothing
			noise = self.m_epsilon(self.policy_noise)
			a_next = self.target_actor_net(next_state) + noise
			a_next = self.action_clip(a_next)

			q_next1 = self.target_critic_net1(next_state, a_next)
			q_next2 = self.target_critic_net2(next_state, a_next)

			# select min q value from q_next1 and q_next2 (double Q learning)
			q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)

		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_net2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		## Delayed Actor(Policy) Updates ##
		if self.total_time_step % self.update_freq == 0:
			## update actor ##
			# actor loss
			# select action a from behavior actor network (a is different from sample transition's action)
			# get Q from behavior critic network, mean Q value -> objective function
			# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			action = self.action_clip(action)
			actor_loss : torch.Tensor = - self.critic_net1(state, action).mean()

			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
		
