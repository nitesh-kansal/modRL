import os, sys
sys.path.append(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
from .utils import *
from collections import deque
import numpy as np
from .ppo import PPO


class A2C_PPO_LOSS(PPO):
    def __init__(self,
                 policy_value_network,
                 optimizer,
                 value_loss_coef=1,
                 n_boot_strap=1,
                 normalize_advantage =True,
                 use_gae_advantage = True,
                 lambda_bootstrap_schedule = lambda old_value, rewards: 0.995*old_value,
                 lambda_bootstrap_start=0.9,
                 ppo_value_epsilon_schedule = lambda old_value, rewards: 0.995*old_value,
                 ppo_value_epsilon_start = 1,
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
        
        super().__init__(None,
                 optimizer,
                 SGD_steps=SGD_steps,
                 ppo_policy_epsilon_schedule=ppo_policy_epsilon_schedule,
                 ppo_policy_epsilon_start=ppo_policy_epsilon_start,
                 entropy_reg_schedule = entropy_reg_schedule,
                 entropy_reg_start = entropy_reg_start,
                 gamma_schedule = gamma_schedule,
                 gamma_start = gamma_start,
                 target_score = target_score,
                 target_score_window = target_score_window,
                 action_transform = action_transform,
                 verbosity = verbosity)
        
        print("inside a2c")
        self.policy_value_network = policy_value_network
        self.value_loss_coef = value_loss_coef
        self.n_boot_strap = n_boot_strap
        self.normalize_advantage = normalize_advantage
        self.use_gae_advantage = use_gae_advantage
        self.lambda_bootstrap_schedule = lambda_bootstrap_schedule
        self.lambda_bootstrap_start = lambda_bootstrap_start
        self.ppo_value_epsilon_schedule = ppo_value_epsilon_schedule
        self.ppo_value_epsilon_start = ppo_value_epsilon_start
    
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device) 
        dist,_ = self.policy_value_network(state_tensor.unsqueeze(0))
        sample = dist.sample()
        sample_local_numpy = sample.detach().cpu().numpy()
        action = self.action_transform(sample_local_numpy)
        return action[0]
    
    def collect_trajectories(self, env, tmax, tmax_for_training):
        states_list = []
        action_samples_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []

        states = env.reset_env()                                # reset the environment    
        scores = np.zeros(env.nagents) 
        t = 0
        while t < tmax:
            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device) 
            states_list.append(states_tensor)                   # storing states
            dists,values = self.policy_value_network(states_tensor)          # getting distribution out of the policy network 
            
            # getting and storing value
            values_list.append(torch.transpose(values,0,1))
            
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
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)[None,:]
            rewards_list.append(rewards_tensor)                # storing rewards

            # storing values for each state
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
            t += 1
        # getting last states value
        last_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        states_list.append(last_states_tensor) 
        _,last_states_values = self.policy_value_network(last_states_tensor)          # getting value out of value network
        last_states_values = last_states_values
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)[None,:]
        last_states_values = torch.transpose(last_states_values,0,1)
        
        # handling terminal states
        last_states_values = torch.where(dones_tensor == 0,last_states_values, torch.zeros_like(last_states_values))
        values_list.append(last_states_values)
        rewards_list.append(last_states_values)
        
        states_list = states_list[:-self.n_boot_strap]
        if self.n_boot_strap > 1:
            action_samples_list = action_samples_list[:-self.n_boot_strap+1]
            log_probs_list = log_probs_list[:-self.n_boot_strap+1]
        
        return torch.cat(states_list[:tmax_for_training],0), torch.cat(action_samples_list[:tmax_for_training],0), \
            torch.cat(rewards_list,0), torch.cat(values_list,0), torch.cat(log_probs_list[:tmax_for_training],0), scores
    
    def future_rewards_and_n_step_bootstrap_advantage(self,rewards, tmax_for_training, V, gamma):
        tot_dis_f_R = self.calculate_future_rewards(rewards, gamma)
        n_step_boot_strap_rewards = tot_dis_f_R[:-self.n_boot_strap] + (gamma**self.n_boot_strap) * (V[self.n_boot_strap:] - tot_dis_f_R[self.n_boot_strap:])
        A = n_step_boot_strap_rewards - V[:-self.n_boot_strap]
        if self.normalize_advantage:
            A_norm = self.normalize_rewards(A)
        tot_dis_f_R = tot_dis_f_R[:-self.n_boot_strap]
        tot_dis_f_R_tensor = tot_dis_f_R[:tmax_for_training].flatten()
        n_step_boot_strap_rewards_tensor = n_step_boot_strap_rewards[:tmax_for_training].flatten()
        A_tensor = A_norm[:tmax_for_training].flatten()
        return A_tensor, tot_dis_f_R_tensor
    
    
    def future_rewards_and_gae_advantage(self,rewards, tmax_for_training, V, discount_rate, lambda_gae):
        tot_dis_f_R = self.calculate_future_rewards(rewards, discount_rate)
        delta = rewards[:-1] + discount_rate*V[1:] - V[:-1]
        A =  self.calculate_future_rewards(delta, discount_rate*lambda_gae)
        if self.normalize_advantage:
            A_norm = self.normalize_rewards(A)
        tot_dis_f_R_tensor = tot_dis_f_R[:tmax_for_training].flatten()
        A_tensor = A_norm[:tmax_for_training].flatten()
        return A_tensor, tot_dis_f_R_tensor
    
    def ppo_value_loss(self, RF, old_value_tensor, new_value_tensor, epsilon_value):
        clipped_new_value = torch.clamp(new_value_tensor, old_value_tensor-epsilon_value, old_value_tensor+epsilon_value)
        clipped_mse_loss = torch.pow(RF - clipped_new_value,2)
        mse_loss = torch.pow(RF - new_value_tensor, 2)
        value_loss = torch.mean(torch.max(mse_loss, clipped_mse_loss))
        return value_loss
    
    def training(self, env, tmax, tmax_for_training, n_episodes):
        
        mean_rewards = []
        mean_rewards_window = deque(maxlen=self.target_score_window)
        
        gamma = self.gamma_start
        epsilon_policy = self.ppo_policy_epsilon_start
        entropy_reg = self.entropy_reg_start
        epsilon_value = self.ppo_value_epsilon_start
        lambda_gae = self.lambda_bootstrap_start
        
        for e in range(n_episodes):
            
            # collect a trajectory
            self.policy_value_network.eval()
            with torch.no_grad():
                states_tensor, action_samples_tensor, rewards_tensor_2d, \
                    old_values_tensor, old_log_prob_tensor, score_numpy =  self.collect_trajectories(env, tmax, tmax_for_training)
                
                # advantage measure
                if self.use_gae_advantage:
                    A_tensor, RF = self.future_rewards_and_gae_advantage(rewards_tensor_2d, tmax_for_training, old_values_tensor, gamma, lambda_gae)
                else:
                    A_tensor, RF = self.future_rewards_and_n_step_bootstrap_advantage(rewards_tensor_2d, tmax_for_training, old_values_tensor, gamma)
                    old_values_tensor = old_values_tensor[:-self.n_boot_strap]
                old_values_tensor = old_values_tensor[:tmax_for_training].flatten()
                        
            # policy update
            self.policy_value_network.train()
            with torch.enable_grad():
                for _ in range(self.SGD_steps):
                    dists,values = self.policy_value_network(states_tensor)
                    new_log_prob_tensor = torch.sum(dists.log_prob(action_samples_tensor),1)
                    policy_loss = self.ppo_policy_loss(A_tensor, old_log_prob_tensor, new_log_prob_tensor, epsilon_policy)
                    entropy_loss = - entropy_reg * (torch.sum(dists.entropy(),1)).mean()
                    total_policy_loss = policy_loss + entropy_loss
                    
                    new_values_tensor = values[:,0]
                    value_loss = self.ppo_value_loss(RF, old_values_tensor, new_values_tensor, epsilon_value)
                    
                    total_loss = total_policy_loss + self.value_loss_coef * value_loss
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            
            # update entropy regularization coefficient
            # this will allow us to reduce entropy regularization in later part of training
            entropy_reg = self.entropy_reg_schedule(entropy_reg,mean_rewards)
            
            # update discount rate of agent
            # this will allow us to focus on longer time rewards in later part of training
            gamma = self.gamma_schedule(gamma,mean_rewards)
            
            # update policy ppo loss epsilon
            # in later part of training we may want to make even finer adjustements
            epsilon_policy = self.ppo_policy_epsilon_schedule(epsilon_policy, mean_rewards)
            
            # update value ppo loss epsilon
            # in later part of training we may want to make even finer adjustements
            epsilon_value = self.ppo_value_epsilon_schedule(epsilon_value, mean_rewards)
            
            # update value GAE lambda
            # as the value estimates becomes better during training we may want to move to 1 step TD error 
            lambda_gae = self.lambda_bootstrap_schedule(lambda_gae, mean_rewards)
            
            # get the average reward of the parallel environments
            mean_of_all_agents = np.mean(score_numpy)
            mean_rewards.append(mean_of_all_agents)
            mean_rewards_window.append(mean_of_all_agents)
            avg_of_last_x_episodes = np.mean(mean_rewards_window)
        
            # display some progress every 20 iterations
            print(f"\r Episode : {e+1}\t Average reward in last 100 episode : {avg_of_last_x_episodes:.2f} epsilon_policy : {epsilon_policy:0.3f} epsilon_value : {epsilon_value:0.3f} lambda_gae : {lambda_gae:0.3f} gamma : {gamma:0.3f} entropy_reg : {entropy_reg:0.3f}",end="")
            if (e+1)%self.verbosity ==0 :
                print(f"\r Episode : {e+1}\t Average reward in last {self.verbosity} episode : {avg_of_last_x_episodes:.2f} epsilon_policy : {epsilon_policy:0.3f} epsilon_value : {epsilon_value:0.3f} lambda_gae : {lambda_gae:0.3f} gamma : {gamma:0.3f} entropy_reg : {entropy_reg:0.3f}")
                
            if (e+1)%(10*self.verbosity) ==0 :
                env.play(self, 1000)

            if avg_of_last_x_episodes >= self.target_score:
                print(f"\nEnvironment solved in episodes = {e+1}")
                break
                
        
        return mean_rewards
    
    
    