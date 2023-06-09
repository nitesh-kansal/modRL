import os, sys
sys.path.append(os.path.abspath(__file__))
import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy

import torch
import torch.nn.functional as F
from .utils import SumTree


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","score"]) 
        self.device = device
                
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""        
        e = self.experience(state, action, reward, next_state, done,1)
        self.memory.insert(e)
    
    def update(self, ids, deltas, alpha):
        for idx,delta in zip(ids,deltas):
            score = np.power(delta, alpha)
            self.memory.update(idx, score)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        ids, experiences = self.memory.sample(self.batch_size)
        N = self.memory.length()
        ISw = np.power(np.array([e.score for e in experiences])*N,-beta)
        ISw = ISw/ISw.max()
        ISw = torch.from_numpy(ISw).float().to(self.device)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones, ids, ISw)
    
    def length(self):
        """Return the current size of internal memory."""
        return self.memory.length()