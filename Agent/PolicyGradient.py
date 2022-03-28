import tensorflow as tf
from tensorflow import keras
import numpy as np

class PolicyGradientAgent():

    def __init__(self, env):
        
        self.env = env

        self.n_max_steps = 99999
        self.discount_rate = 0.99
        self.learning_rate=0.02

        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.loss_fn = keras.losses.categorical_crossentropy

        self.model = keras.models.Sequential()

    def build_model(self, input_shape, output_num):

        self.model = keras.models.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=[input_shape]),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(output_num, activation="softmax"),
    ])

    def play_one_step(self, obs):

        with tf.GradientTape() as tape:
            prob = self.model(obs[np.newaxis])[0]
            action = np.random.choice(range(len(prob.numpy().ravel())), p=prob.numpy().ravel())
            y_target = tf.constant(np.array([(1 if i==action else 0) for i in range(len(prob))])) # one-hot

            loss = tf.reduce_mean(self.loss_fn(y_target, prob))
        grads = tape.gradient(loss, self.model.trainable_variables)
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, grads

    def play_multiple_episodes(self, n_episodes):

        self.all_rewards = []
        self.all_grads = []

        for episode in range(n_episodes):
            current_rewards = []
            current_grads = []
            obs = self.env.reset()
            for step in range(self.n_max_steps):
                obs, reward, done, grads = self.play_one_step(obs)
                current_rewards.append(reward)
                current_grads.append(grads)
                if sum(current_rewards) < -250:
                    done = True
                if done:
                    break
            self.all_rewards.append(current_rewards)
            self.all_grads.append(current_grads)

        return self.all_rewards, self.all_grads

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
    