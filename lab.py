# from Agent.PolicyGradient import *
from a2c import *
from statistics import mean, median, pstdev
import gym

common_env = [ "CartPole-v1", "LunarLander-v2" ]
env = gym.make(common_env[0])

agent = ActorCriticAgent(env)

n_iterations = 200
n_episodes_per_update = 10

mean_rewards_over_all_iter = []

for iteration in range(n_iterations):

    agent.play_multiple_episodes(n_episodes_per_update)
    agent.updatePolicy()

    mean_rewards_over_all_iter.append(mean(agent.rewards_per_episode_over_iter))

    print("*   Iteration:", iteration+1)
    print("* Total Games:", (iteration+1)*n_episodes_per_update)
    print("*")
    print("*    Mean:", mean(agent.rewards_per_episode_over_iter))
    print("*  Median:", median(agent.rewards_per_episode_over_iter))
    print("*  StdDev:", pstdev(agent.rewards_per_episode_over_iter))
    print("*     Min:", min(agent.rewards_per_episode_over_iter))
    print("*     Max:", max(agent.rewards_per_episode_over_iter))
    print("==================")

env.close()

import matplotlib.pyplot as plt
    
plt.plot([i*n_episodes_per_update+n_episodes_per_update for i in range(len(mean_rewards_over_all_iter))], mean_rewards_over_all_iter)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Avg. Training Rewards")
plt.savefig('output/images/plot.png')
    