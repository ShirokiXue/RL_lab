"""
ref: https://github.com/MorvanZhou/pytorch-A3C
"""
import numpy as np
import gym
import os
from utils import v_wrap, set_init, record
from shared_adam import SharedAdam
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

os.environ["OMP_NUM_THREADS"] = "1"

DISCOUNT_RATE = 0.9
UPDATE_GLOBAL_ITER = 200
MAX_STEP = 500
MAX_EP = 1000
LEARNING_RATE = 1e-3

ENV = 'CartPole-v1'
# ENV = "LunarLander-v2"
env = gym.make(ENV)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_layer = nn.Linear(64, a_dim)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))

        action_prob = F.softmax(self.action_layer(x), dim=-1)
        state_values = self.value_layer(x)

        return action_prob, state_values

class Agent():
    def __init__(self):

        self.gnet = Net(N_S, N_A)        # global network
        self.gnet.share_memory()         # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.gnet.parameters(), lr=LEARNING_RATE)      # global optimizer
        self.global_ep, self.global_ep_r, self.res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    def train(self):

        # parallel training
        workers = [Worker(self, i) for i in range(mp.cpu_count()//2-1)]
        [w.start() for w in workers]
        self.res = []                    # record episode reward to plot
        while True:
            r = self.res_queue.get()
            if r is not None:
                self.res.append(r)
            else:
                break
        [w.join() for w in workers]

    def show_result(self):

        import matplotlib.pyplot as plt
        plt.plot(self.res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()

class Worker(mp.Process):
    def __init__(self, agent, name):

        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = agent.global_ep, agent.global_ep_r, agent.res_queue
        self.gnet, self.opt = agent.gnet, agent.opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make(ENV)

    def select_and_step_action(self, state):
        state  = torch.from_numpy(state).float()
        probs, state_value = self.lnet(state)

        m = Categorical(probs)
        action = m.sample()

        self.SavedActions.append( SavedAction(m.log_prob(action), state_value) )

        state, reward, done, info = self.env.step(action.item())
        self.rewards.append(reward)

        return state, reward, done, info

    def discount_rewards(self, rewards):

        discounted = np.array(rewards)

        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * DISCOUNT_RATE

        return discounted

    def discount_and_normalize_rewards(self):

        discounted_rewards = self.discount_rewards(self.rewards)
                          
        reward_mean = discounted_rewards.mean()
        reward_std = discounted_rewards.std()

        final_rewards = (discounted_rewards - reward_mean) / (reward_std + eps)

        return final_rewards

    def push_and_pull(self, opt, lnet, gnet):

        optimizer = opt
        policy_losses = []
        value_losses  = []
        
        for (log_prob, value), reward in zip(self.SavedActions, self.rewards):

            advantage = reward - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward])))

        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp.grad
        optimizer.step()

        lnet.load_state_dict(gnet.state_dict())

    def run(self):

        
        while self.g_ep.value < MAX_EP:

            self.SavedActions = []
            self.rewards = []
            state = self.env.reset()
            ep_r = 0.
            step = 1

            while True:

                state, r, done, _ = self.select_and_step_action(state)
                if done: r = -1
                ep_r += r

                if step >= MAX_STEP or ep_r <= -100: done = True

                if step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    self.rewards = self.discount_and_normalize_rewards()
                    self.push_and_pull(self.opt, self.lnet, self.gnet)
                    self.SavedActions, self.rewards = [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                step += 1

        self.res_queue.put(None)


if __name__ == "__main__":

    agent = Agent()
    agent.train()
    agent.show_result()
    