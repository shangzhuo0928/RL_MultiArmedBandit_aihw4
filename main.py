from BanditEnv import BanditEnv
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

k = 10
e = [0, 0.01, 0.1]

env=BanditEnv(k)

x = np.arange(1, 1001)
y1, y2 = [], []
#part3 env means不變 資料1000筆x
for p in range(3):
    env.reset()
    agent = Agent(k, e[p])
    agent.reset()
    avry = np.zeros(1000)
    pery = np.zeros(1000)
    for j in range(2000):
        t = 0 
        env.reset()
        agent.reset()
        for i in range(1000):
            opt_a = np.argmax(env.means)
            action = agent.select_action()
            reward=env.step(action)
            agent.update_q(action, reward)
            if action == opt_a:
                pery[i] += 1

        action_history,reward_history = env.export_history()
        avry += np.array(reward_history)/2000
    pery /= 20 
    
    y1.append(avry)
    y2.append(pery)

for i in range(3):
    plt.plot(x, y1[i], label=f'ε={e[i]}')
plt.title('Average Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.savefig('part3_reward.png')

plt.clf()

for i in range(3):
    plt.plot(x, y2[i], label=f'ε={e[i]}')
plt.title('average optimal action selection rate')
plt.xlabel('Steps')
plt.ylabel('%')
plt.legend()
plt.savefig('part3_per.png')

plt.clf()

env=BanditEnv(k, False)
y1, y2 = [], []
x = np.arange(1, 10001)
#part5 env means變 資料10000筆x
for p in range(3):
    env.reset()
    agent = Agent(k, e[p])
    agent.reset()
    avry = np.zeros(10000)
    pery = np.zeros(10000)
    for j in range(2000):
        t = 0 
        env.reset()
        agent.reset()
        for i in range(10000):
            opt_a = np.argmax(env.means)
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)
            if action == opt_a:
                pery[i] += 1

        action_history,reward_history = env.export_history()
        avry += np.array(reward_history)/2000
    pery /= 20 

    y1.append(avry)
    y2.append(pery)

for i in range(3):
    plt.plot(x, y1[i], label=f'ε={e[i]}')
plt.title('Average Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.savefig('part5_reward.png')

plt.clf()

for i in range(3):
    plt.plot(x, y2[i], label=f'ε={e[i]}')
plt.title(' average optimal action selection rate')
plt.xlabel('Steps')
plt.ylabel('%')
plt.legend()
plt.savefig('part5_per.png')

plt.clf()

y1, y2 = [], []
#part7 env means變 資料10000筆x 
for p in range(3): 
    env.reset()
    agent = Agent(k, e[p], 0.1) #agent a固定0.1
    agent.reset()
    avry = np.zeros(10000)
    pery = np.zeros(10000)
    for j in range(2000):
        t = 0 
        env.reset()
        agent.reset()
        for i in range(10000):
            opt_a = np.argmax(env.means)
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)
            if action == opt_a:
                pery[i] += 1

        action_history,reward_history = env.export_history()
        avry += np.array(reward_history)/2000
    pery /= 20 

    y1.append(avry)
    y2.append(pery)

for i in range(3):
    plt.plot(x, y1[i], label=f'ε={e[i]}')
plt.title('Average Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.savefig('part7_reward.png')

plt.clf()

for i in range(3):
    plt.plot(x, y2[i], label=f'ε={e[i]}')
plt.title(' average optimal action selection rate')
plt.xlabel('Steps')
plt.ylabel('%')
plt.legend()
plt.savefig('part7_per.png')