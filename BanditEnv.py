import numpy as np

class BanditEnv():
    def __init__(self, k, stationary = True):
        self.k = k
        self.var = 1
        self.s = stationary
        self.means = np.random.standard_normal(size=self.k)
        self.action_history = []
        self.reward_history = []

    def reset(self):
        self.means = np.random.standard_normal(size=self.k)
        self.action_history = []
        self.reward_history = []
        
    def step(self, action):
        if not self.s:
            walks = np.random.normal(0, 0.01, size=self.k)
            self.means += walks 
        if 0 <= action < self.k:
            r = np.random.normal(self.means[action], self.var)
            self.action_history.append(action)
            self.reward_history.append(r)
            return r

    def export_history(self):
        return self.action_history, self.reward_history