from unityagents import UnityEnvironment

class UnityBananaCollector:
    def __init__(self, envpath, is_state_pixeled=False, train_mode=True):
        # create the environment
        self.env = UnityEnvironment(file_name=envpath)
        
        # Extracting the primary acting brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        
        # reset the environment
        self.env_info = self.env.reset(train_mode=train_mode)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        print('Number of actions:', self.action_size)

        # examine the state space
        print('States have shape (raw):', self.env_info.vector_observations[0].shape)
        
        # initial state modifying state
        initial_state = self.convert_state(self.env_info.vector_observations[0] , is_state_pixeled)
        self.state_size = initial_state.shape
        print('States have shape (after conversion):', self.state_size)
        
        # params
        self.is_state_pixeled = is_state_pixeled
        self.train_mode = train_mode
        
    def convert_state(self, state, is_state_pixeled):
        if is_state_pixeled:
            state = np.transpose(state, (0, 3, 1, 2))
        else:
            state = state[None,:]
        return state
    
    def reset_env(self):
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        initial_state = self.convert_state(self.env_info.vector_observations[0] , self.is_state_pixeled) 
        return initial_state
        
    def take_action(self,action):
        self.env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
        next_state = self.convert_state(self.env_info.vector_observations[0] , self.is_state_pixeled)   # get the next state
        reward = self.env_info.rewards[0]                   # get the reward
        done = self.env_info.local_done[0]                  # see if episode has finished
        return next_state, reward, done
    
    def exit_env():
        self.env.close()