from unityagents import UnityEnvironment

class ReacherEnvironment:
    def __init__(self, envpath, convert_fn, train_mode=True ):
        # create the environment
        self.env = UnityEnvironment(file_name=envpath)
        
        # Extracting the primary acting brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        # reset the environment
        self.env_info = self.env.reset(train_mode=train_mode)[self.brain_name]

        # number of agents in the environment
        self.nagents = len(self.env_info.agents)
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        print('Number of actions:', self.action_size)

        # examine the state space 
        states = self.env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        
        # params
        self.train_mode = train_mode
        self.convert_fn = convert_fn
    
    def reset_env(self):
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        initial_state = self.convert_fn(self.env_info.vector_observations)
        return initial_state
        
    def take_action(self,actions):
        self.env_info = self.env.step(actions)[self.brain_name]        # send the action to the environment
        next_states = self.convert_fn(self.env_info.vector_observations)   # get the next state
        rewards = self.env_info.rewards                   # get the reward
        dones = self.env_info.local_done                  # see if episode has finished
        return next_states, rewards, dones
    
    def exit_env():
        self.env.close()