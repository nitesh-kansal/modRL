import random
import torch
import numpy as np
from collections import deque
from dqnagent import DQNAgent

def watch_trained_agent(agent, env):
    print("******************* watch trained agent in action **********************")
    state = env.reset_env()
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps=0)          # select an action
        next_state, reward, done = env.take_action(action)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    print("Score of the sample episode after training: {}".format(score))


def dqn_training_loop(env, qnetwork, score_benchmark, window_size, params):
    
    """Deep Q-Learning.
    
    qnetwork: pytorch model
    score_benchmark: At what score is the environment considered to be solved
    window_size: For how many consecutive episodes to do moving average of scores for evaluation agent score
    
    params: dict
    ==================
        n_episodes (int) : maximum number of training episodes
        max_t (int) : maximum number of timesteps per episode
        policy_type ("e_greedy","annealed_softmax") : choice of policy used for selecting actions while collecting transitions
        
        Parameters useful in case of epsilon greedy action selection
        eps_start (float) : starting value of epsilon, for epsilon-greedy action selection
        eps_end (float) : minimum value of epsilon
        eps_decay (float) : multiplicative factor (per episode) for decreasing epsilon
        
        Parameters useful in case of annealed softmax action selection
        t_start (float) : starting value of temperature in for annealed softmax action selection
        t_end (float) : minimum value of temperature
        t_decay (float) : multiplicative factor (per episode) for decreasing temperature
        
        Parameters for controlling dependence of sampling probability on TD-error in Prioritized Experience Replay
        alphaf (float) : Final value of alpha parameter in Prioritized Experience Replay 
        alpha0 (float) : Initial value of alpha parameter in Prioritized Experience Replay
        nsteps_alpha (int) : Number of episodes in which to linearly change alpha from alpha0 to alphaf
        
        Parameters for controlling Importance Sampling weight (ISw) in Prioritized Experience Replay
        betaf (float) : Final value of beta parameter in Prioritized Experience Replay 
        beta0 (float) : Initial value of beta parameter in Prioritized Experience Replay
        nsteps_beta (int) : Number of episodes in which to linearly change beta from beta0 to betaf
        
        DQN Update Parameters
        LR (float) : Learning Rate for update of DQN weights
        BUFFER_SIZE (int) : Size of the Replay Buffer
        TAU (float) : Fraction of primary network weights to be copied over to the target network after each parameter update step
                θ_target = τ*θ_primary + (1 - τ)*θ_target
        BATCH_SIZE (int) : Size of the sample to be selected at random from the Replay Buffer at each update step 
        UPDATE_EVERY (int) : Number of actions (or transitions to be recorded) to be taken before making any update to DQN weights
        SAMPLE_FREQ (int) : Number of batch sampling and DQN weight update steps to be carried out during the update step
        GAMMA (float) : Discount Factor
        IS_DDQN (bool) : Whether to enable the Double DQN improvement or continue with basic DQN
        MEMORY_TYPE ("prioritized","normal"): Whether to go with prioritized memory buffer or uniform
    """
    # defining DQN Agent
    agent = DQNAgent(qnetwork, params)
    
    optimal_selection_p = []
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=window_size)  # last 100 scores
    avg_i = []
    eps = params["eps_start"]                    # initialize epsilon
    t = params["t_start"]
    for i_episode in range(1, params["n_episodes"]+1):
        # alpha value
        malpha = (params["alphaf"] - params["alpha0"])/(params["nsteps_alpha"] - 1)
        alpha = malpha*(i_episode - 1) + params["alpha0"]
        
        # beta value
        mbeta = (params["betaf"] - params["beta0"])/(params["nsteps_beta"] - 1)
        beta = mbeta*(i_episode - 1) + params["beta0"]
            
        score = 0                                          # initialize the score        
        i = 0
        
        state = env.reset_env()
        while i < params["max_t"]:
            if params["policy_type"] == "e_greedy":
                action = agent.act_e_greedy(state, eps)       # select an action
            elif params["policy_type"] == "annealed_softmax":
                action = agent.act_annealed_softmax(state, t)
            next_state, reward, done = env.take_action(action)
            agent.step(state, action, reward, next_state, done, alpha = alpha, beta = beta)
            score += reward                                # update the score
            state = next_state  # roll over the state to next time step
            i += 1
            if done:                                       # exit loop if episode finished
                break 
        avg_i.append(i)
        optimal_selection_p.append(float(agent.optimal_selection)/agent.action_selection)
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(params["eps_end"], params["eps_decay"]*eps) # decrease epsilon
        t = max(params["t_end"], params["t_decay"]*t) # decrease temperature
        print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Episode Len : {}'.format(i_episode, np.mean(scores_window), np.mean(avg_i)), end="")
        if i_episode % window_size == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Episode Len : {}'.format(i_episode, np.mean(scores_window), np.mean(avg_i)))
        if np.mean(scores_window)>=score_benchmark:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-window_size, np.mean(scores_window)))
            print('Saving DQN weights in ./checkpoint.pth')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores, agent, optimal_selection_p