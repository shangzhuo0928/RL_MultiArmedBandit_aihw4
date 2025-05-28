# RL_MultiArmedBandit_aihw4
There is k arms.
The reward distribution for each arm be a Gaussian distribution with variance 1 and a randomly generated mean generated from standard normal distribution.
Compare the greedy method (𝜖 = 0) and two 𝜖-greedy methods (𝜖 = 0.1 and 𝜖 = 0.01) result from 3 kinds of situation.

## 1. Stationary environment, sample-average method agent:

Experiments 2,000 times independently, and run the agent for 1000 steps.

## 2. Non-stationary environment, sample-average method agent:
   
(Independent random walks by adding a
 normally distributed increment with mean 0 and standard deviation 0.01 to the mean reward of all action.)

Experiments 2,000 times independently, and run the agent for 10000 steps.

## 2. Non-stationary environment, the fixed step-size (a) agent:

Experiments 2,000 times independently, and run the agent for 10000 steps.

 
