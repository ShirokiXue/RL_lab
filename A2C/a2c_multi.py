from concurrent.futures import thread
import gym
import numpy as np
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

# ENV = "CartPole-v1"
ENV = "LunarLander-v2"
N_ITERS = 1000
N_EP = 8
MAX_STEPS = 200
DISCOUNT_RATE = 0.99
LEARNING_RATE = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
eps = np.finfo(np.float32).eps.item()
# n_cpu = mp.cpu_count()//2-1
# n_cpu = 4

class Memory():

    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def discount_and_normalize_rewards(self):

        discounted = np.array(self.rewards)
        dones   = self.dones

        for i in range( len(self.rewards)-2, -1, -1 ):

            if not dones[i]:
                discounted[i] += discounted[i+1] * DISCOUNT_RATE

        _mean = discounted.mean()
        _std = discounted.std()

        self.rewards = [(reward - _mean) / (_std + eps) for reward in discounted]
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)

    def _extend(self, m):

        self.log_probs += m.log_probs
        self.values += m.values
        self.rewards += m.rewards
        self.dones += m.dones
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

class PolicyNet(nn.Module):

    def __init__(self, input_shape: int, output_shape: int):

        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)

        self.action_layer = nn.Linear(64, output_shape)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        action_prob = F.softmax(self.action_layer(x), dim=-1)
        state_values = self.value_layer(x)

        return action_prob, state_values

class A2C():
    
    def __init__(self):
        
        self.env        = gym.make(ENV)
        self.input_n    = self.env.observation_space.shape[0]
        self.output_n   = self.env.action_space.n
        print(f"intput_n: {self.input_n}, output_n: {self.output_n}")
        self.memory = Memory()
        self.memory_queue = mp.Queue()

        self.net = PolicyNet(self.input_n, self.output_n)
        self.net.share_memory()
        self.optimizer  = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss()

        self.net.to(device)

    def run_multiple_episodes(self, n_episodes):

        iter_rewards = []

        workers = [Worker(self.net, self.memory, iter_rewards, j) for j in range(N_EP)]
        [w.start() for w in workers]
        [w.join() for w in workers]

        return iter_rewards

    def updatePolicy(self):

        optimizer = self.optimizer
        self.memory.discount_and_normalize_rewards()
        policy_losses = []
        value_losses  = []
        
        for log_prob, value, reward, _ in self.memory._zip():

            advantage = reward - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss( value, torch.tensor([reward]).to(device)))

        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()

        self.memory.clear()

    def save_model(self):

        torch.save(self.net.state_dict(), f"./model/{ENV}_model_weights.pth")

class Worker(threading.Thread):
    def __init__(self, net, g_memory, iter_rewards, name):

        threading.Thread.__init__(self)

        self.name = 'w%02i' % name
        self.env = gym.make(ENV)
        self.net = net

        self.g_memory = g_memory
        self.iter_rewards = iter_rewards
        self.memory = Memory()

    def step_and_add_action(self, state):

        state  = torch.from_numpy(state).float().to(device)
        probs, state_value = self.net(state)

        m = Categorical(probs)
        action = m.sample()

        state, reward, done, info = self.env.step(action.item())

        self.memory.add( m.log_prob(action), state_value, reward, done )

        return state, reward, done, info

    def run(self):

        ep_reward = 0
        state = self.env.reset()

        for t in range(MAX_STEPS):

            state, reward, done, _ = self.step_and_add_action(state)
            ep_reward += reward
            
            if done:
                self.env.close()
                break
        
        self.g_memory._extend(self.memory)
        self.iter_rewards.append(ep_reward)


if __name__ == "__main__":

    agent = A2C()
    record = []

    for iteration in range(N_ITERS):

        print(f"Iter: {iteration+1}, ", end='')
        iter_rewards = agent.run_multiple_episodes(N_EP)
        record.append(iter_rewards)
        print(f"Mean_Reward: {np.array(iter_rewards).mean()}")
        agent.updatePolicy()
        

    agent.save_model()


    import matplotlib.pyplot as plt
    
    x = [(i+1)*N_EP for i in range(N_ITERS)]
    y = [np.array(iter_rewards).mean() for iter_rewards in record]
    plt.plot(x, y)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Avg. Training Rewards")
    plt.savefig('./image/plot.png')
        
