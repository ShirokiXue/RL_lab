import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_shape, 16)
        self.fc2 = nn.Linear(16, output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=-1)
        return x

class PolicyGradientAgent():
    def __init__(self, env):
        
        self.env = env

        # n = 44
        # self.env.seed(n)
        # np.random.seed(n)
        # torch.manual_seed(n)

        self.n_max_steps = 999
        self.discount_rate = 0.99
        learning_rate=0.01

        self.action_space = env.action_space.n

        self.policy_net = PolicyNet(env.observation_space.shape[0], self.action_space)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def play_one_step(self, state):

        state  = torch.from_numpy(state).float()
        probs   = self.policy_net(state)
        action = np.random.choice(range(self.action_space), p=probs.detach().numpy())
        
        next_state, reward, done, _ = self.env.step(action)
        target = action

        return state, next_state, target, reward, done

    def play_multiple_episodes(self, n_episodes):

        self.all_states = []
        self.all_targets = []
        self.all_rewards = []

        for episode in range(n_episodes):

            current_states = []
            current_targets = []
            current_rewards = []
            state = self.env.reset()

            for step in range(self.n_max_steps):

                state, next_state, target, reward, done = self.play_one_step(state)
                current_targets.append(target)
                current_rewards.append(reward)
                current_states.append(state)
                state = next_state

                if sum(current_rewards) < -250:
                    done = True
                if done:
                    break
            
            # print(torch.stack(current_probs).type(torch.float32))
            # print(torch.tensor(current_targets, dtype=torch.int64))
            # self.all_probs.append( torch.stack(current_probs).type(torch.float32) )
            self.all_targets.append( torch.tensor(current_targets, dtype=torch.int64) )
            self.all_states.append(current_states)
            self.all_rewards.append(current_rewards)

            self.rewards_per_episode_over_iter = [sum(rewards) for rewards in self.all_rewards]

        return self.all_states, self.all_targets, self.all_rewards

    def discount_rewards(self, rewards):

        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * self.discount_rate

        return discounted

    def discount_and_normalize_rewards(self):

        all_discounted_rewards = [self.discount_rewards(rewards)
                                for rewards in self.all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()

        self.all_final_rewards = [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

        return self.all_final_rewards

    def updatePolicy(self):

        all_final_rewards = self.discount_and_normalize_rewards()
        
        for states, targets, rewards in zip(self.all_states, self.all_targets, all_final_rewards):

            self.optimizer.zero_grad()

            for state, target, reward in zip(states, targets, rewards):

                prob   = self.policy_net(state)
                loss = self.loss_fn(prob, target) * reward
                loss.backward()

            self.optimizer.step()