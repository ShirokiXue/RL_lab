import gym
import numpy as np
from collections import namedtuple
from pickletools import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ENV = "CartPole-v1"
N_ITERS = 50
N_EP = 16
MAX_STEPS = 200
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.01

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
eps = np.finfo(np.float32).eps.item()

# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
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

    def discount_rewards(self):

        discounted = np.array(self.rewards)
        dones   = self.dones

        for i in range( len(self.rewards)-2, -1, -1 ):

            if not dones[i]:
                discounted[i] += discounted[i+1] * DISCOUNT_RATE

        self.rewards = discounted

    def normalize_rewards(self):

        _mean = self.rewards.mean()
        _std =  self.rewards.std()

        self.rewards = [(reward - _mean) / (_std + eps) for reward in self.rewards]
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
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

        self.fc1 = nn.Linear(input_shape, 32)

        self.action_layer = nn.Linear(32, output_shape)
        self.value_layer = nn.Linear(32, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        action_prob = F.softmax(self.action_layer(x), dim=-1)
        state_values = self.value_layer(x)

        return action_prob, state_values

class ActorCritic():
    
    def __init__(self):
        
        self.env        = gym.make(ENV)
        self.input_n    = self.env.observation_space.shape[0]
        self.output_n   = self.env.action_space.n
        self.memory     = Memory()
        self.record     = []

        self.policy_net = PolicyNet(self.input_n, self.output_n)
        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        self.policy_net.to(device)

    def step_and_add_action(self, state):

        state  = torch.from_numpy(state).float().to(device)
        probs, state_value = self.policy_net(state)

        m = Categorical(probs)
        action = m.sample()

        state, reward, done, info = self.env.step(action.item())

        self.memory.add( m.log_prob(action), state_value, reward, done )

        return state, reward, done, info

    def run_multiple_episodes(self, n_episodes):

        iter_rewards = []

        for i_episode in range(n_episodes):

            ep_reward = 0
            state = self.env.reset()

            for t in range(MAX_STEPS):

                state, reward, done, _ = self.step_and_add_action(state)
                ep_reward += reward

                if done:
                    break

            iter_rewards.append(ep_reward)
            
        mean_reward = np.array(iter_rewards).mean()
        return mean_reward

    def updatePolicy(self):

        optimizer = self.optimizer
        self.memory.discount_rewards()
        self.memory.normalize_rewards()
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


if __name__ == "__main__":

    agent = ActorCritic()

    for iteration in range(N_ITERS):

        print(f"Iter: {iteration+1}, ", end='')
        mean_reward = agent.run_multiple_episodes(N_EP)
        agent.updatePolicy()
        print(f"Mean_Reward: {mean_reward}")

    pass
        
        
