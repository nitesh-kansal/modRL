import os, sys
import gymnasium as gym
import collections
import numpy as np
import cv2
sys.path.append(os.path.abspath(__file__))

class MujocoGymNumStateContActions:
    def __init__(self, envname, n_parallel_envs = 20,n_consecutive_states = 4):
        # saving environment name
        self.envname = envname
        
        # create the environment
        self.envs = gym.vector.make(envname, num_envs=n_parallel_envs) #, render_mode="human")

        # number of actions
        print('action space:', self.envs.action_space)

        # examine the state space
        print('raw state space :', self.envs.observation_space)
        
        # number of agents
        self.nagents = n_parallel_envs
    
    def reset_env(self):
        initial_state, info = self.envs.reset()
        return initial_state
        
    def take_action(self, actions):
        nextstates, rewards, terminated, truncated, info = self.envs.step(actions)
        dones = terminated | truncated
        return nextstates, rewards, dones
    
    def exit_env(self):
        self.envs.close()
        
    def play(self, agent, tmax):
        env = gym.make(self.envname, render_mode="human")
        state, info = env.reset()
        score = 0
        i = 0
        while True:
            action = agent.act(state)
            nextstate, reward, terminated, truncated, info = env.step(action)
            score += (reward - score)/(i+1)
            print(f"\r score @ {i} = {score}", end="")
            state = nextstate
            if terminated or truncated or (i >= tmax):
                break
            i += 1
        env.close()
        