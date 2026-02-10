from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
class DubinsRA_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.high = np.array([
            1., 1., 1., 1.
        ])
        self.low = np.array([
            -1., -1., -1., -1.
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # joint action space
        
        self.constraint = [-0.5, 0.0, 0.25] # constraint: [x, y, r]
        self.goal = [0.5, 0.0, 0.25] # constraint: [x, y, r]
        self.u_max = 1.25

        
    def step(self, action):
        # action is in -1, 1. Scale by self.u_max

        self.state[0] = self.state[0] + self.dt * self.state[3]
        self.state[1] = self.state[1] + self.dt * self.state[2]

        theta = np.arctan2(self.state[2], self.state[3]) # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)


        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        # l(x) = -((x-xg)^2 + (y-yg)^2 - r^2)

        rew = -((self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2 - self.goal[2]**2)
        
        # g(x) = (x-x0)^2 + (y-y0)^2 - r^2
        constraint = ((self.state[0]-self.constraint[0])**2 + (self.state[1]-self.constraint[1])**2 - self.constraint[2]**2)
        
        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True

        if constraint < 0:
            terminated = True
        #if rew > 0:
        #    terminated = True
        info = {"constraint": np.array([10*constraint]).astype(np.float32)}
        return self.state.astype(np.float32), 10*rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2*np.pi)

            self.state = np.random.uniform(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], size=(4,))
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)
        else:
            self.state = initial_state     
        self.state = self.state.astype(np.float32)   
        return self.state, {}
