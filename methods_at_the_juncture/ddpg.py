import os, sys
sys.path.append(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
from ..utils import *
from collections import deque
import numpy as np
from copy import deepcopy
from ..memory.memory import PrioritizedReplayBuffer
import random

from os import *
from sys import *
from collections import *
from math import *

class DDPG:
    def __init__(self,
                 policy_network,
                 q_network,
                 optimizer_policy,
                 optimizer_q,
                 action_transform = lambda x: x,
                 wait_before_start = 256,
                 memory_buffer = 10000,
                 batch_size = 64,
                 gamma_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 gamma_start = 0.985,
                 update_every = 4,
                 num_updates = 4,
                 tau = 1e-3,
                 target_score_window = 100,
                 target_score=30,
                 memory_beta_start = 10,
                 memory_beta_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 memory_alpha_start = 10,
                 memory_alpha_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 sigma_noise_start = 10,
                 sigma_noise_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 device = get_device(),
                 verbosity = 100
                ):
        
        self.device = device
        self.policy_network = policy_network
        self.q_network = q_network
        self.target_policy_network = deepcopy(policy_network)
        self.target_q_network = deepcopy(q_network)
        self.optimizer_policy = optimizer_policy
        self.optimizer_q = optimizer_q
        self.action_transform = action_transform
        self.wait_before_start = wait_before_start
        self.memory_buffer = memory_buffer
        self.batch_size = batch_size
        self.gamma_schedule = gamma_schedule
        self.gamma_start = gamma_start
        self.update_every = update_every
        self.num_updates = num_updates
        self.tau = tau
        self.target_score_window = target_score_window
        self.target_score = target_score
        self.sigma_noise_start = sigma_noise_start
        self.sigma_noise_schedule = sigma_noise_schedule
        self.memory_beta_start = memory_beta_start
        self.memory_beta_schedule = memory_beta_schedule
        self.memory_alpha_start = memory_alpha_start
        self.memory_alpha_schedule = memory_alpha_schedule
        self.memory = PrioritizedReplayBuffer(memory_buffer, batch_size, device)
        self.verbosity = verbosity
        
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) 
        action = self.policy_network(state_tensor.unsqueeze(0))
        action_local_numpy = action.detach().cpu().numpy()[0,:]
        action = self.action_transform(action_local_numpy)
        return action
    
    def noisy_action(self, state, sigma):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) 
        self.policy_network.eval()
        with torch.no_grad():
            action_value = self.policy_network(state_tensor)
        action_value = action_value + torch.randn(action_value.shape)*sigma
        action_value = action_value.detach().cpu().numpy()[0,:]
        action = self.action_transform(action_value)
        return action
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def update(self, alpha, beta, gamma):
        for i in range(self.num_updates):
            experiences = self.memory.sample(beta)
            states, actions, rewards, next_states, dones, ids, ISw = experiences

            self.target_policy_network.eval()
            self.target_q_network.eval()
            with torch.no_grad():
                optimal_actions = self.target_policy_network(next_states)
                q_target = self.target_q_network({'states': next_states, 'actions':optimal_actions})
                q_target = rewards + gamma*q_target*(1.-dones)
                
            self.q_network.train()
            self.policy_network.train()
            with torch.enable_grad():
                self.optimizer_q.zero_grad()
                q_pred = self.q_network({'states':states, 'actions': actions})
                loss = (ISw * (q_pred - q_target) ** 2).mean()
                loss.backward()
                self.optimizer_q.step()
                
                self.optimizer_policy.zero_grad()
                new_actions = self.policy_network(states)
                neg_qval = -self.q_network({'states':states, 'actions': new_actions})
                neg_qval.backward()
                self.optimizer_policy.step()
            
            
            # ********************************* PRIORITIZED REPLAY **************************#
            new_scores = torch.abs(q_pred - q_target).squeeze(1).detach().cpu().data.numpy() + 1e-2
            self.memory.update(ids,new_scores,alpha)
            # ********************************* ****************** **************************#
        self.soft_update(self.policy_network, self.target_policy_network)
        self.soft_update(self.q_network, self.target_q_network)
            
    
    def training(self, env, tmax, n_episodes):
        
        mean_rewards = []
        mean_rewards_window = deque(maxlen=self.target_score_window)
        
        gamma = self.gamma_start
        sigma = self.sigma_noise_start
        beta = self.memory_beta_start
        alpha = self.memory_alpha_start
        
        time_steps = 0
        for e in range(n_episodes):
            state = env.reset_env()                                # reset the environment    
            score = 0
            episode_steps = 0
            while episode_steps <= tmax:
                action = self.noisy_action(state, sigma)
                next_state, reward, done = env.take_action(action)
                self.memory.add(state, action, reward, next_state, done)
                score += reward
                
                if (((time_steps + 1) % self.wait_steps) == 0) and (self.memory.length() >= self.batch_size):
                    self.update(alpha, beta, gamma)
                    
                episode_steps += 1
                time_steps += 1
                
                if done:
                    break
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            gamma = self.gamma_schedule(gamma,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            sigma = self.sigma_noise_schedule(sigma,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            beta = self.memory_beta_schedule(beta,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            alpha = self.memory_alpha_schedule(alpha,mean_rewards)
            
            # get the average reward of the parallel environments
            mean_rewards.append(score)
            mean_rewards_window.append(score)
            avg_of_last_x_episodes = np.mean(mean_rewards_window)
            # display some progress every iteration
            print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} sigma : {sigma:.3f} gamma : {gamma:.3f} beta : {beta:.3f} alpha : {alpha:.3f}",end="")
            if (e+1) % self.verbosity == 0 :
                print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} sigma : {sigma:.3f} gamma : {gamma:.3f} beta : {beta:.3f} alpha : {alpha:.3f}")

            if avg_of_last_x_episodes >= self.target_score:
                print(f"\nEnvironment solved in episodes = {e+1}")
                break
        
        return mean_rewards
            
            