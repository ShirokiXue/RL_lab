"""
@author: ShirokiXue

ref: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
"""

from pickletools import optimize
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class PolicyNet(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):

        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_shape, 16)
        self.fc2 = nn.Linear(16, output_shape)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)

        return x

class PolicyGradientAgent():
    def __init__(self, env, n_max_steps = 999,
                            discount_rate = 0.99,
                            learning_rate=0.01):
        
        self.env = env

        # n = 543
        # self.env.seed(n)
        # torch.manual_seed(n)

        self.n_max_steps   = n_max_steps
        self.discount_rate = discount_rate
        
        self.action_space = env.action_space.n

        self.policy_net = PolicyNet(env.observation_space.shape[0], self.action_space)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_and_step_action(self, state):

        state  = torch.from_numpy(state).float()
        probs   = self.policy_net(state)

        m = Categorical(probs)
        action = m.sample()

        self.SavedActions.append(SavedAction(m.log_prob(action),0))

        state, reward, done, info = self.env.step(action.item())
        self.rewards.append(reward)

        return state, reward, done, info

    def play_multiple_episodes(self, n_episodes):

        self.all_rewards = []
        self.all_SavedActions = []

        for i_episode in range(n_episodes):

            self.SavedActions = []
            self.rewards = []
            state = self.env.reset()

            for t in range(self.n_max_steps):

                state, _, done, _ = self.select_and_step_action(state)

                if done:
                    break
            
            self.all_rewards.append(self.rewards)
            self.all_SavedActions.append(self.SavedActions)
            self.rewards_per_episode_over_iter = [sum(rewards) for rewards in self.all_rewards]

    def discount_rewards(self, rewards):

        discounted = np.array(rewards)
        discount_rate = self.discount_rate

        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_rate

        return discounted

    def discount_and_normalize_rewards(self):

        all_discounted_rewards = [self.discount_rewards(rewards)
                                for rewards in self.all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()

        all_final_rewards = [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

        return all_final_rewards

    def updatePolicy(self):

        all_final_rewards = self.discount_and_normalize_rewards()
        policy_losses = []
        
        for saved_actions, rewards in zip(self.all_SavedActions, all_final_rewards):

            for (log_prob, _), reward in zip(saved_actions, rewards):
                policy_losses.append(-log_prob * reward)

        optimizer = self.optimizer
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss.backward()
        optimizer.step()

    def save_model( self, filename = "model", path = "output/model" ):
        
        torch.save( self.policy_net.state_dict(), f"{path}/{filename}.pth")
