import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
from statistics import mean, median, pstdev

from Network.PolicyGradient import *

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

env = gym.make("LunarLander-v2")
env.seed(42)

n_iterations = 10
n_episodes_per_update = 20

agent = PolicyGradientAgent(env)
agent.build_model(env.observation_space.shape[0], env.action_space.n)

all_mean_rewards = []

for iteration in range(n_iterations):

    all_rewards, all_grads = agent.play_multiple_episodes(n_episodes_per_update)
    rewards_per_episode = [sum(rewards) for rewards in all_rewards]
    all_mean_rewards.append(mean(rewards_per_episode))

    print("==================")
    print("*   Iteration:", iteration+1)
    print("* Total Games:", (iteration+1)*n_episodes_per_update)
    print("*")
    print("*    Mean:", mean(rewards_per_episode))
    print("*  Median:", median(rewards_per_episode))
    print("*  StdDev:", pstdev(rewards_per_episode))
    print("*     Min:", min(rewards_per_episode))
    print("*     Max:", max(rewards_per_episode))
    print("==================")

    all_final_rewards = agent.discount_and_normalize_rewards()
    all_mean_grads = []

    for var_index in range(len(agent.model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)

    agent.optimizer.apply_gradients(zip(all_mean_grads, agent.model.trainable_variables))

env.close()
agent.model.save(f'output\model_lunarlander.h5')

import matplotlib.pyplot as plt
    
plt.plot([i*n_episodes_per_update+n_episodes_per_update for i in range(len(all_mean_rewards))], all_mean_rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Avg. Training Rewards")
plt.savefig('output/images/plot.png')
