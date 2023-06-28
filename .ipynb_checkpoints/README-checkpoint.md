[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

### Introduction

This repository provides python implementation of Deep Q-Learning Methods for solving various RL based problems in which the states, actions, rewards are related by a transition probability distribution which is stationary in time (i.e. not changing over time and hence there is a single optimal policy to be found). The repository can currently be used to solve (maximize total expected reward)  environments (simulated or real) where the states are representated as n dimensional continuous features and actions take any of the known 'k' discrete values. The methods implemented in this repository are proven to perform at near human levels on various Atari games.

You can find easy to use implementation of vanilla Deep Q-Network (DQN) algorithm (from the early DeepMind [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) and some of the early notable improvements over the basic method e.g.
- Double DQN model: [click_here](https://arxiv.org/pdf/1509.06461.pdf).
- Prioritized Replay: [click_here](https://arxiv.org/pdf/1511.05952.pdf).
- Dueling Architecture DQN: [click_here](https://arxiv.org/abs/1511.06581).
- Boltzman or Softmax Action Selection: [click_here](https://proceedings.neurips.cc/paper/2017/file/b299ad862b6f12cb57679f0538eca514-Paper.pdf)
- More Improvement coming

There are example notebooks in repository in which we have show how to use the repository in order to solve some of the challenging environments available in OpenAI Gym and Unity ML-Agents. Furthermore, in this repository you can easily choose the improvements which you want to keep and switch the unwanted once off while training the Agent.

### The Banana Collector Environment

The task is to train an Agent to navigate in a large square world and collect banana present at various location in it. The environment is as shown in the clip below.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The same environment is made available by Udacity in two different versions, one in which the states are in raw format i.e. (1 X 84 X 84 X 3) dimensional numpy array representing the intantenous frames visible to the agent, while in the other more simpler version the agent is made available with states which are 37 dimensional representations of the frames. Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of 13+ over 100 consecutive episodes.

### Installation Instructions

#### Setting up python environment

1. Create (and activate) a new environment with Python 3.6.
	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Clone the repository.  Then, install several dependencies.
```bash
git clone https://github.com/nitesh-kansal/Deep_q_learning.git
cd Deep_q_learning/python
pip install .
```
3. Installing pytorch. Please visit [official page](https://pytorch.org/) of pytorch to find the appropriate conda command for your system.

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

#### Downloading Banana Collector Environment

1. To download and solve the easier environment in which states are in transformed form (37 dimensional), choose from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.
    
2. To download and solve the harder environment in which states correspond to raw pixels (1 X 84 X 84 X 3 dimensional rgb images), choose from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

3. Place the downloaded environment file in the root directory of the repository i.e. in  `Deep_q_learning/` folder, and unzip (or decompress) the file. 

4. Note, you may have to give extra permissions to the environment to access inputs from sources other than the keyboard and produce outputs on destinations other than the usual.

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