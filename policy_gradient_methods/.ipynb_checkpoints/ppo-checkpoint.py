import os, sys
sys.path.append(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
from .reinforce import REINFORCE
from utils import *

class PPO(REINFORCE):
    def __init__(self,
                 policy_network,
                 optimizer,
                 SGD_steps=4,
                 ppo_policy_epsilon_schedule=lambda old_value, rewards: 0.995*old_value,
                 ppo_policy_epsilon_start=0.1,
                 entropy_reg_schedule = lambda old_value, rewards: 0.995*old_value,
                 entropy_reg_start=0.01,
                 gamma_schedule = lambda old_value, rewards: 1. - 0.995*(1. - old_value),
                 gamma_start = 0.985,
                 target_score=30,
                 target_score_window=100,
                 action_transform = lambda x: x,
                 verbosity = 50):
        
        super().__init__(policy_network,
                 optimizer,
                 entropy_reg_schedule = entropy_reg_schedule,
                 entropy_reg_start = entropy_reg_start,
                 gamma_schedule = gamma_schedule,
                 gamma_start = gamma_start,
                 target_score = target_score,
                 target_score_window = target_score_window,
                 action_transform = action_transform,
                 loss_type = "ratio",
                 verbosity = verbosity)
        
        self.SGD_steps = SGD_steps
        self.ppo_policy_epsilon_schedule = ppo_policy_epsilon_schedule
        self.ppo_policy_epsilon_start = ppo_policy_epsilon_start
    
    def ppo_policy_loss(self, advantage, old_log_prob_tensor, new_log_prob_tensor, epsilon):
        log_prob_diff = new_log_prob_tensor - old_log_prob_tensor
        prob_ratio = torch.exp(log_prob_diff)
        clipped_prob_ratio = torch.clamp(prob_ratio, 1.-epsilon, 1.+epsilon)
        surr_clipped_R = (torch.min(advantage*prob_ratio, advantage*clipped_prob_ratio)).mean()
        return -surr_clipped_R
    
    def training(self, env, tmax, tmax_for_training, n_episodes):
        
        mean_rewards = []
        mean_rewards_window = deque(maxlen=self.target_score_window)
        
        gamma = self.gamma_start
        epsilon = self.ppo_policy_epsilon_start
        entropy_reg = self.entropy_reg_start
        for e in range(n_episodes):
            
            # collect a trajectory
            self.policy_network.eval()
            with torch.no_grad():
                states_tensor, action_samples_tensor, rewards_tensor_2d, \
                    old_log_prob_tensor, score_numpy =  self.collect_trajectories(env, tmax, tmax_for_training)
            
                # advantage measure from reinforce class
                advantage = self.mean_std_advantage(rewards_tensor_2d, tmax_for_training, gamma)
            
            # policy update
            self.policy_network.train()
            with torch.enable_grad():
                for _ in range(self.SGD_steps):
                    dists,_ = self.policy_network(states_tensor)
                    new_log_prob_tensor = torch.sum(dists.log_prob(action_samples_tensor),1)
                    policy_loss = self.ppo_policy_loss(advantage, old_log_prob_tensor, new_log_prob_tensor, epsilon)
                    entropy_loss = - entropy_reg * (torch.sum(dists.entropy(),1)).mean()
                    total_loss = policy_loss + entropy_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            
            # update entropy regularization coefficient
            # this will allow us to reduce entropy regularization in later part of training
            entropy_reg = self.entropy_reg_schedule(entropy_reg,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            gamma = self.gamma_schedule(gamma,mean_rewards)
            
            # update ppo loss epsilon
            # in later part of training, we may want to make even finer adjustements
            epsilon = self.ppo_policy_epsilon_schedule(epsilon, mean_rewards)
            
            # get the average reward of the parallel environments
            mean_of_all_agents = np.mean(score_numpy)
            mean_rewards.append(mean_of_all_agents)
            mean_rewards_window.append(mean_of_all_agents)
            avg_of_last_x_episodes = np.mean(mean_rewards_window)
            # display some progress every iteration
            print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} epsilon_policy : {epsilon:.3f} gamma : {gamma:.3f} entropy_reg : {entropy_reg:.3f}",end="")
            if (e+1)%self.verbosity == 0 :
                print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} epsilon_policy : {epsilon:.3f} gamma : {gamma:.3f} entropy_reg : {entropy_reg:.3f}")

            if avg_of_last_x_episodes >= self.target_score:
                print(f"\nEnvironment solved in episodes = {e+1}")
                break
        
        return mean_rewards