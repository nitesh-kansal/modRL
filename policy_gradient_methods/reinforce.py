import os, sys
sys.path.append(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
from .utils import *
from collections import deque
import numpy as np


class REINFORCE:
    def __init__(self,
                 policy_network,
                 optimizer, 
                 entropy_reg_schedule = lambda old_value, rewards: 0.995*old_value,
                 entropy_reg_start=0.01,
                 gamma_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 gamma_start = 0.985,
                 target_score=30,
                 target_score_window=100,
                 action_transform = lambda x: x,
                 loss_type = "ratio",
                 verbosity = 50):
        
        self.device = get_device()
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.entropy_reg_schedule = entropy_reg_schedule
        self.entropy_reg_start = entropy_reg_start
        self.gamma_schedule = gamma_schedule
        self.gamma_start = gamma_start
        self.target_score = target_score
        self.target_score_window = target_score_window
        self.action_transform = action_transform
        self.loss_type = loss_type
        self.verbosity = verbosity
    
    def collect_trajectories(self, env, tmax, tmax_for_training):
        states_list = []
        action_samples_list = []
        rewards_list = []
        log_probs_list = []

        states = env.reset_env()                                # reset the environment    
        scores = np.zeros(env.nagents) 
        t = 0
        while t <= tmax:
            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device) 
            states_list.append(states_tensor)                   # storing states
            dists,_ = self.policy_network(states_tensor)          # getting distribution out of the policy network
            
            # getting log prob of sampled actions
            samples = dists.sample()                            # sampling from distribution
            log_probs = torch.sum(dists.log_prob(samples),1)
            log_probs_list.append(log_probs)                    # storing model probs

            # converting samples to action space (e.g. [0,1] to [-h,h])
            action_samples_list.append(samples)                 # storing action samples tensor
            samples_local_numpy = samples.detach().cpu().numpy()     # detached, brings to cpu and convert to numpy
            actions = self.action_transform(samples_local_numpy)

            # generating rewards by interacting with environment
            next_states, rewards, dones = env.take_action(actions)
            rewards_tensor = torch.tensor([rewards], dtype=torch.float32, device=self.device)
            rewards_list.append(rewards_tensor)                # storing rewards

            # storing values for each state
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
            t += 1
        return torch.cat(states_list[:tmax_for_training],0), torch.cat(action_samples_list[:tmax_for_training],0), \
            torch.cat(rewards_list,0), torch.cat(log_probs_list[:tmax_for_training],0), scores
    
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) 
        dist,_ = self.policy_network(state_tensor.unsqueeze(0))
        sample = dist.sample()
        sample_local_numpy = sample.detach().cpu().numpy()
        action = self.action_transform(sample_local_numpy)
        return action[0]
    
    def calculate_future_rewards(self, rewards, discount_rate):
        tot_dis_f_R = torch.zeros_like(rewards)
        tot_dis_f_R[rewards.shape[0]-1] = rewards[rewards.shape[0]-1]
        for i in range(rewards.shape[0]-2,-1,-1):
            tot_dis_f_R[i] = rewards[i] + discount_rate*tot_dis_f_R[i+1]
        return tot_dis_f_R
    
    def normalize_rewards(self, R):
        mean = R.mean(dim=1, keepdims=True)
        std = R.std(dim=1, keepdims=True)
        R = (R - mean)/(std + 1e-10)
        return R
    
    def mean_std_advantage(self, rewards, tmax_for_training, discount_rate):
        # calculating future cumulative rewards and normalize
        tot_dis_f_R = self.calculate_future_rewards(rewards, discount_rate)
        norm_tot_dis_f_R = self.normalize_rewards(tot_dis_f_R)[:tmax_for_training]
        return norm_tot_dis_f_R.flatten()
        
    def basic_policy_loss(self, advantage, old_log_prob_tensor, new_log_prob_tensor):
        if self.loss_type == "ratio":
            log_prob_diff = new_log_prob_tensor - old_log_prob_tensor
            prob_ratio = torch.exp(log_prob_diff)
            ExpR = (advantage*prob_ratio).mean()
        else:
            ExpR = (advantage*new_log_prob_tensor).mean()
        return -ExpR
            
    def training(self, env, tmax, tmax_for_training, n_episodes):
        
        mean_rewards = []
        mean_rewards_window = deque(maxlen=self.target_score_window)
        
        entropy_reg = self.entropy_reg_start
        gamma = self.gamma_start
        for e in range(n_episodes):
            
            # collect a trajectory
            self.policy_network.eval()
            with torch.no_grad():
                states_tensor, action_samples_tensor, rewards_tensor_2d, \
                    old_log_prob_tensor, score_numpy =  self.collect_trajectories(env, tmax, tmax_for_training)
            
                # advantage measure
                advantage = self.mean_std_advantage(rewards_tensor_2d, tmax_for_training, gamma)
            
            # policy update
            self.policy_network.train()
            with torch.enable_grad():
                dists,_ = self.policy_network(states_tensor)
                new_log_prob_tensor = torch.sum(dists.log_prob(action_samples_tensor),1)
                policy_loss = self.basic_policy_loss(advantage, old_log_prob_tensor, new_log_prob_tensor)
                entropy_loss = - entropy_reg * (torch.sum(dists.entropy(),1)).mean()
                total_loss = policy_loss + entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step() 

            # get the average reward of the parallel environments
            mean_of_all_agents = np.mean(score_numpy)
            mean_rewards.append(mean_of_all_agents)
            mean_rewards_window.append(mean_of_all_agents)
            avg_of_last_x_episodes = np.mean(mean_rewards_window)
            
            # update entropy regularization coefficient
            # this will allow us to reduce entropy regularization in later part of training
            entropy_reg = self.entropy_reg_schedule(entropy_reg,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            gamma = self.gamma_schedule(gamma,mean_rewards)
        
            # display some progress every 20 iterations
            print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} gamma : {gamma:0.3f} entropy_reg : {entropy_reg:0.3f}",end="")
            if (e+1)%self.verbosity ==0 :
                print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} gamma : {gamma:0.3f} entropy_reg : {entropy_reg:0.3f}")

            if avg_of_last_x_episodes >= self.target_score:
                print(f"\nEnvironment solved in episodes = {e+1}")
                break
        
        return mean_rewards