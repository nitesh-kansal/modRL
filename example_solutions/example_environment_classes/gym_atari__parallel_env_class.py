import gymnasium as gym
import collections
import numpy as np
import cv2

class OpenAIGym:
    def __init__(self, envname,n_parallel_envs = 4,n_consecutive_states = 4):
        # create the environment
        gym.vector.make(envname, num_envs=n_parallel_envs)

        # number of actions
        self.action_size = self.env.action_space.n
        print('Number of actions:', self.action_size)

        # examine the state space
        self.raw_state_shape = self.env.observation_space.shape
        print('images have shape :', self.raw_state_shape)
        
        # running states
        self.images = collections.deque(maxlen=n_consecutive_states)
        self.n_consecutive_states = n_consecutive_states
    
    def generate_state(self):
        concatenated_images = np.concatenate(tuple(self.images),axis=0)
        return concatenated_images[None,:,:,:]
        
    def pre_process_image(self, image):
        resized_image = cv2.resize(image, dsize=(92, 92), interpolation=cv2.INTER_CUBIC)
        transposed_image = np.transpose(resized_image, (2, 0, 1))
        return transposed_image
    
    def add_image_to_queue(self,image):
        transposed_image = self.pre_process_image(image)
        self.images.append(transposed_image)
    
    def reset_env(self):
        initial_image, info = self.env.reset()
        for i in range(self.n_consecutive_states):
            self.add_image_to_queue(initial_image)
        initial_state = self.generate_state()
        return initial_state
        
    def take_action(self, action):
        next_image, reward, terminated, truncated, info = self.env.step(action)
        self.add_image_to_queue(next_image)
        next_state = self.generate_state()
        return next_state, reward, (terminated or truncated)
    
    def exit_env():
        self.env.close()