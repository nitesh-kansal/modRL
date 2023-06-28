import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import multiprocessing as mp

class BBOAgent:
    """Interacts with the environment and optimizes policy using black box optimization methods."""
    
    def __init__(self, env, policy_network ,params, max_episode_len=1000):
        """
        Initialize an Agent object.
        
        Params
        ======
            policy_network (DNN model): pytorch model
            params (dict) : all parameters required for choosing the optimization method
                restart_prob (float) : probability of restarting after an unsuccessful evaluation (f(x) did not increase)
                
                min_steps_before_restart (int) : minimum number of unsuccessful evaluation before restarting 
                
                n_neighbours (int) : Number of neighbours to sample before each update
                            n_neighbours = 1 gives 'Basic Hill Climbing'
                            
                k_top (int) : Retain top k for averaging
                
                boltzman_t_fn (function) : Function for variation of boltzman_t over updates
                            fn(t_old, update_i) -> t_new
                            Weight given to each candidate is proportional to np.exp(f(x)/boltzman_t)
                                boltzman_t = inf means simply averaging (Cross Entropy Method) 
                                boltzman_t = 0 means max value picked only (Steepest Hill Climbing)
                                boltzman_t in (0, inf) means Evolution Strategies
                            
                sigma_fn (function) : Function for sigma variation in Simulated Annealing
                            fn(sigma_old, update_i, has_reward_increased) -> sigma_new
                            fn(sigma_old,...) = constant => no annealing
                            fn(sigma_old,update_i,...) = f(sigma_old, update_i) => simulated annealing
                            fn(sigma_old,update_i,has_reward_increased) = f(sigma_old, has_reward_increased) => adaptive annealing
                            
                nthread (int) : number of parallel function evaluations
                            <=0 means equal to cpu count
        """
        self.env
        self.policy_network = policy_network
        self.params = params
        
    
    def evaluate(self, weights):
        """
            Evaluates current policy network defined by the weight argument 
        """
        
        for param, w in zip(self.policy_network.parameters(), weights):
            param.data = nn.parameter.Parameter(w)
        
        self.policy_network.eval()
        ###################### episode ##########################
        i = 0
        state = env.reset_env()
        score = 0
        while i < self.params["max_t"]:
            state = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                if self.params["action_space"] == "discrete":
                    action = np.argmax(self.policy_network(state).cpu().data.numpy()[0,:])
                elif self.params["action_space"] == "continous":
                    action = self.policy_network(state).cpu().data.numpy()[0,0]
                    
            next_state, reward, done = env.take_action(action)
            score += reward
            state = next_state
            i += 1
            if done:
                break
        return i,score
    
    def initialize(self, mean=0., std=1.):
        """
            Random initialization of weights
        """
        weights = []
        for param in self.policy_network.parameters():
            weights.append(mean + std*torch.randn(param.shape))
        return weights
    
    def perturb_weights(self, weights, sigma):
        """
            Does random perturbations to weights
        """
        return [w + sigma * torch.randn(w.shape) for w in weights]
    
    def 
            
    
    def single_step(self, weights_in, sigma_in, reward_in, nthread, update_i):
        """
            Executes single step of optimization
        """
        candidates = [perturb_weights(weights_in, sigma_in) for i in range(self.params["n_neighbours"])]
        pool = mp.Pool(nthread)
        rewards = pool.map(self.evaluate, candidates)
        mx_reward = max(rewards)
        has_reward_increased = (mx_reward > reward_in)
        if has_reward_increased:
            
        else:
            
    
    def black_box_optimization_loop(self):
        nthread = max([min([self.params["nthread"],mp.cpu_count()]),0])
        nthread = mp.cpu_count() if (nthread == 0) else nthread
        weights_init = initialize()
        
        
            