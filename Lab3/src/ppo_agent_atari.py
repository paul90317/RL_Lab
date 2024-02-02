import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym
import math
from gym.wrappers import FrameStack, RecordVideo
import random
from torch.distributions import Categorical
import sys

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		
		### TODO ###
		# initialize test_env
		# self.test_env = ???

		self.env = FrameStack(gym.make(config['env_id']),4)
		self.test_env = FrameStack(gym.make(config['env_id'],render_mode='rgb_array'),4)
		self.test_env = RecordVideo(self.test_env, 'video',episode_trigger=lambda x:True)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

		if len(sys.argv)==3:
			self.load(sys.argv[1])
			self.total_time_step = int(sys.argv[2])
		elif len(sys.argv)!=1:
			exit('error argv number')

		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, action_prob from net

		observation = torch.tensor(observation, device = self.device, dtype= torch.float32)
		with torch.no_grad():
			logits, value = self.net(observation)
			dist = Categorical(logits=logits)
			if eval:
				action = torch.argmax(logits,dim=1,keepdim=True)
			else:
				action = dist.sample().unsqueeze(1)

				if random.random() < (math.exp(-self.total_time_step/400000) - 0.1):
					action[0][0] = 1
			action_prob = dist.probs.gather(1,action)
			return action.squeeze(), value.squeeze(), action_prob.squeeze(),
	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				logits, value = self.net(ob_train_batch)
				dist = Categorical(logits=logits)

				# calculate policy loss
				# ratio = ???
				# surrogate_loss = ???
				action_prob = dist.probs.gather(1,ac_train_batch).squeeze()
				ratio = action_prob / (logp_pi_train_batch + 0.00001)
				surrogate_loss = - torch.min(ratio * adv_train_batch, torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch).mean()

				# calculate value loss
				# value_criterion = nn.MSELoss()
				# v_loss = value_criterion(...)
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value.squeeze(), return_train_batch)
				
				# calculate total loss
				# loss = surrogate_loss + self.value_coefficient * v_loss - (10 * math.exp(-self.total_time_step/100000) + self.entropy_coefficient) * entropy
				entropy = dist.entropy().squeeze().mean()
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				# nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



