import numpy as np

class Agent():
    def __init__(self, k, epsilon, alpha = None):
        self.k = k
        self.e = epsilon
        self.alpha = alpha
        self.q_v = np.zeros(self.k)
        self.a_c = np.zeros(self.k)

    def reset(self):
        self.q_values = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
    
    def  update_q(self, action, reward):
        self.a_c[action] += 1
        if self.alpha == None:
            a = 1/self.a_c[action]
        else:
            a = self.alpha
        
        self.q_v[action] += a* (reward - self.q_v[action])
    
    def select_action(self):
        if np.random.rand() <= self.e:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_v)
