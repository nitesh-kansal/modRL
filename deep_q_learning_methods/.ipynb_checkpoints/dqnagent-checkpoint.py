import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
from memory.memory import PrioritizedReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, qnetwork, params):
        """Initialize an Agent object.
        
        Params
        ======
            qnetwork (DNN model): pytorch model
        """
        self.params = params

        # Q-Network
        self.qnetwork_local = qnetwork
        self.qnetwork_target = deepcopy(qnetwork)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params["LR"])

        # Replay memory
        self.memory = PrioritizedReplayBuffer(params["BUFFER_SIZE"], params["BATCH_SIZE"])
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # optimal action selected
        self.optimal_selection = 0
        self.action_selection = 0
    
    def step(self, state, action, reward, next_state, done, alpha=0., beta=0.):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params["UPDATE_EVERY"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.length() > self.params["BATCH_SIZE"]:
                self.learn(self.params["GAMMA"],self.params["SAMPLE_FREQ"], alpha, beta)

    def act_annealed_softmax(self, state, t=0.):
        """Returns actions for given state as per value proportional policy.
            i.e. 
            
            q(a,s) =  exp[Q(a,s)^(1/t)]
            
            p(a/s) = (q(a/s)/sum[q(a/s)])
            
            t is temperature coefficient and is gradually decreased
                        
        Params
        ======
            state (array_like): current state
            t (float): temperature for annealed softmax action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Boltzman Distribution based action selection
        action_values = action_values.cpu().data.numpy()[0,:]
        policy = action_values/t
        policy -= policy.max()
        policy = np.exp(policy)
        policy = policy/policy.sum()
        action = np.random.choice(action_values.shape[0], p = policy)
        if action == np.argmax(policy):
            self.optimal_selection += 1
        self.action_selection += 1
        return action

    def act_e_greedy(self, state, eps=0.):
        """Returns actions for given state as per normal epsilon-greedy policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        # Epsilon-greedy action selection
        action_values = action_values.cpu().data.numpy()[0,:]
        policy = np.ones(shape=action_values.shape)*eps/action_values.shape[0]
        policy[np.argmax(action_values)] += 1. - eps
        action = np.random.choice(action_values.shape[0], p = policy)
        if action == np.argmax(policy):
            self.optimal_selection += 1
        self.action_selection += 1
        return action

    def learn(self, gamma, sample_freq, alpha, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float) : discount factor
            alpha (float) : power in prioritization
            beta (float)  : power in importance sampling using for prioritization
        """
        
        for i in range(sample_freq):
            experiences = self.memory.sample(beta)
            states, actions, rewards, next_states, dones, ids, ISw = experiences

            # implemented Double DQN
            self.qnetwork_target.eval()
            self.qnetwork_local.eval()
            with torch.no_grad():
                if self.params["IS_DDQN"]:
                    best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                    q_target = self.qnetwork_target(next_states).gather(1, best_actions)
                else:
                    q_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                q_target = rewards + gamma*q_target*(1.-dones)
                
            self.qnetwork_local.train()
            q_pred = self.qnetwork_local(states).gather(1, actions)
            loss = (ISw * (q_pred - q_target) ** 2).mean()
                        
            # ********************************* PRIORITIZED REPLAY **************************#
            if self.params["MEMORY_TYPE"] == "prioritized":
                # updating priority of a sample
                new_scores = torch.abs(q_pred - q_target).squeeze(1).detach().cpu().data.numpy() + 1e-2
                self.memory.update(ids,new_scores,alpha)
            # ********************************* ****************** **************************#
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params["TAU"]) 
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
