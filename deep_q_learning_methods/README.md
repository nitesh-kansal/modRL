### Introduction

This repository provides python implementation of Deep Q-Learning and Policy Gradient Methods for solving various RL based problems in which the states, actions, rewards are related by a transition probability distribution which is stationary in time (i.e. not changing over time and hence there is a single optimal policy to be found).

### Deep Query Network (DQN)
You can find easy to use implementation of vanilla Deep Q-Network (DQN) algorithm (from the early DeepMind [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) and some of the early notable improvements over the basic method e.g.
- Double DQN model: [click_here](https://arxiv.org/pdf/1509.06461.pdf).
- Prioritized Replay: [click_here](https://arxiv.org/pdf/1511.05952.pdf).
- Dueling Architecture DQN: [click_here](https://arxiv.org/abs/1511.06581).
- Boltzman or Softmax Action Selection: [click_here](https://proceedings.neurips.cc/paper/2017/file/b299ad862b6f12cb57679f0538eca514-Paper.pdf)
- More Improvements coming

### Policy Gradients


There are example notebooks in repository in which we have shown how to use the repository in order to solve some of the challenging environments available in OpenAI Gym and Unity ML-Agents. Furthermore, in this repository you can easily choose the improvements which you want to keep and switch the unwanted once off while training the Agent.

### Installation Instructions

#### Setting up python environment

1. Clone the repository.  Then, install several dependencies.
```bash
git clone https://github.com/nitesh-kansal/modRL.git
```
2. Installing pytorch. Please visit [official page](https://pytorch.org/) of pytorch to find the appropriate conda command for your system.

#### How to use the Implementation?

Define the pytorch DNN model class as defined in 'qmodel.py' file and initialize the model.
<pre><code>
from qmodel import QNetwork
model = QNetwork(37,4,1234,False)
</code></pre>

The training method can be used with any environment of your choice. Just Define the environment wrapper class as defined in 'unity_banana_collector.py' file for Unity Banana Collector Environment. Basically you have to replicate all the functions inside the class in the file i.e.
1. **`__init__`** : sets up the environment.
2. **`reset_env`** : resets the environment for new episode and produces the initial state. Note the state need to be of shape (1, state_size), basically unsqueeze the zeroth dimension e.g. if state_size is (84,84,3) then output (1,84,84,3). 
3. **`take_action`** : need to take the action decided by the agent and produce the next_state, reward and done (if the episode is finished or not).
4. **`exit_env`** : stops the running environment.

Then just initialize the class in your notebook or anywhere else.
<pre><code>
from unity_banana_collector import UnityBananaCollector
env = UnityBananaCollector("Banana.app")
</code></pre>

Then finally for running a DQN loop you have to create a python dictionary of parameter values required for training of a DQNagent and pass it into the training function together with few more parameters i.e. environment object, model object, solving criteria and score averaging window.

<pre><code>
from dqnLoop import dqn_training_loop, watch_trained_agent
params = {}
params["n_episodes"] = 600         # maximum number of training episodes [INT]
params["max_t"] = 1000             # maximum number of timesteps per episode [INT]
params["policy_type"] = "e_greedy" # whether to choose epsilon-greedy ("e_greedy") policy or boltzman softmax ("annealing_softmax") for taking actions

# Parameters useful in case of epsilon greedy action selection
params["eps_start"] = 1.           # starting value of epsilon, for epsilon-greedy action selection [FLOAT] [0 to 1]
params["eps_end"] = 0.01           # minimum value of epsilon [FLOAT] [0 to 1]
params["eps_decay"] = 0.995        # multiplicative factor (per episode) for decreasing epsilon [FLOAT] [0 to 1]

# Parameters useful in case of annealed softmax action selection
params["t_start"] = 5.           # starting value of temperature, for epsilon-greedy action selection [FLOAT]
params["t_end"] = 0.005           # minimum value of temperature [FLOAT]
params["t_decay"] = 0.95        # multiplicative factor (per episode) for decreasing temperature [FLOAT] [0 to 1]

# Parameters for controlling dependence of sampling probability on TD-error in Prioritized Experience Replay, these will not be use if params["MEMORY_TYPE"] = "normal".
params["alpha0"] = 0.0             # Initial value of alpha parameter in Prioritized Experience Replay [FLOAT] [0 to 1]
params["alphaf"] = 0.25            # Final value of alpha parameter in Prioritized Experience Replay [FLOAT] [0 to 1]
params["nsteps_alpha"] = 250       # Number of episodes in which to linearly change alpha from alpha0 to alphaf [INT]

# Parameters for controlling Importance Sampling weight (ISw) in Prioritized Experience Replay, these will not be use if params["MEMORY_TYPE"] = "normal".
params["beta0"] = 0.2              # Final value of beta parameter in Prioritized Experience Replay [FLOAT] [0 to 1]
params["betaf"] = 1                # Initial value of beta parameter in Prioritized Experience Replay [FLOAT] [0 to 1]
params["nsteps_beta"] = 500        # Number of episodes in which to linearly change beta from beta0 to betaf [INT]

#DQN Update Parameters
params["LR"] = 1e-3                # Learning Rate for update of DQN weights [FLOAT] [0 to 1]
params["BUFFER_SIZE"] = 100000     # Size of the Replay Buffer [INT]
params["TAU"] = 1e-2               # Fraction of primary network weights to be copied over to the target network after each parameter update step 
                                        # θ_target = τ*θ_primary + (1 - τ)*θ_target [FLOAT] [0 to 1]
params["BATCH_SIZE"] = 64          # Size of the sample to be selected at random from the Replay Buffer at each update step [INT]
params["UPDATE_EVERY"] = 4         # Number of actions (or transitions to be recorded) to be taken before making any update to DQN weights [INT]
params["SAMPLE_FREQ"] = 1          # Number of batch sampling and DQN weight update steps to be carried out during the update step [INT]
params["GAMMA"] = 0.9              # Discount Factor [FLOAT] [0 to 1]
params["IS_DDQN"] = False          # Whether to enable the Double DQN improvement or continue with basic DQN [BOOL]
params["MEMORY_TYPE"] = "normal"   # Whether to go with prioritized memory buffer or uniform ["normal","prioritized"]

output_scores, agent, optimal_selection_p  =
        dqn_training_loop(env,     # environment class object
            model,                 # model class object
            13.,                   # at what score is the environment considered solved
            100,                   # moving window size over which to average the episode scores
            params)                # the learning agents specifications
score_plot(output_scores, optimal_selection_p)  # this is a python utility that we have created for ploting the scores, you can find this in Report.ipynb notebook.
</code></pre>

DQN weights will be present in the checkpoint.pth file in the same directory where you executed the above code.
