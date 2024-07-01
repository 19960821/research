import random
import numpy as np
import gym
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from env_DQN import *
'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env
'''


class DQN:
    def __init__(
            self, 
            env, 
            memory_cap=100000,
            time_steps=1,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=32,
            tau=0.125
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.state_shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
        
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.summaries = {}
        self.actorloss = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_shape[0]*self.time_steps, activation="relu"))
        model.add(Dense(300, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_dim))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state[0]

    def act(self, test=False): 
        states = self.stored_states.reshape((1, self.state_shape[0]*self.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            #一个在0-self.env.action_dim-1之间的随机数
            # print("-------探索----------")
            return np.random.randint(0, self.env.action_dim)
            # return self.env.sample()  # 随机返回一个action索引，用于探索
        # print("----------利用----------")
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state):
        self.memory.append([state, action, reward, new_state])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        states, action, reward, new_states = map(np.asarray, zip(*samples))
        batch_states = np.array(states).reshape(self.batch_size, -1)
        batch_new_states = np.array(new_states).reshape(self.batch_size, -1)
        batch_target = self.target_model.predict(batch_states)
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        batch_target[range(self.batch_size), action] = reward + q_future * self.gamma
        hist = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        self.summaries['loss'] = np.mean(hist.history['loss'])
        self.actorloss.append(hist.history['loss'])

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.model.save(fn)

    def load_model(self, fn):
        # load model from .h5 file
        self.model = tf.keras.models.load_model(fn)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=50, max_steps=50):
        ep_reward = []
        ep_opt = []
        ep_min_objective = []
        ep_fin_objective = []
        ep_action = []
        ep_num = []

        opt = -10
        min_objective = 0.6
        # opt_action = np.ones(3*self.env.user_num)

        episode, steps, total_reward = 0, 0, 0

        cur_state_O, cur_state_N = self.env.reset()
        self.update_states(cur_state_N)

        while episode < max_episodes:
            opt_num = 0
            while steps<max_steps:
                action = self.act()  # model determine action, states taken from self.stored_states

                #done, next_state, reward, objective, flag, available_num
                done, new_state_O, new_state_N, reward, objective, flag, available_num = self.env.step(cur_state_O, action)  # perform action on env

                if done:
                    break

                prev_stored_states = self.stored_states
                self.update_states(new_state_N)  # update stored states
                self.remember(prev_stored_states, action, reward, self.stored_states)  # add to memory
                self.replay()  # iterates default (prediction) model through memory replay
                self.target_update()  # iterates target model

                cur_state_O = new_state_O
                cur_state_N = new_state_N
                total_reward += reward
                steps += 1


                if flag and available_num==self.env.user_num:# and available_num>=0.5*env.user_num
                    if reward>opt:
                        opt = reward
                    if objective<min_objective:
                        min_objective=objective
                        # opt_action = action
                        opt_num = available_num


            self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
            print("episode {}: {} reward".format(episode, total_reward))

            ep_reward.append(total_reward)
            ep_fin_objective.append(objective)
            ep_opt.append(opt)
            ep_min_objective.append(min_objective)
            # ep_action.append(opt_action)
            ep_num.append(opt_num)

            cur_state_O, cur_state_N = self.env.reset()
            self.update_states(cur_state_N)  # update stored states
            episode += 1
            steps = 0
            total_reward = 0 

        return ep_opt, ep_min_objective
        plt.plot([i+1 for i in range(max_episodes)], ep_reward)
        plt.xlabel("episode")
        plt.ylabel("rewards")
        plt.show()

        plt.plot([i+1 for i in range(max_episodes)], ep_opt, label = "DQN")
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend()
        plt.show()

        plt.plot([i+1 for i in range(max_episodes)], ep_fin_objective)
        plt.xlabel("episode")
        plt.ylabel("ep_fin_objective")
        plt.show()

        plt.plot([i+1 for i in range(max_episodes)], ep_min_objective, label = "DQN")
        plt.xlabel("episode")
        plt.ylabel("objective")
        plt.legend()
        plt.show()

        plt.plot([i+1 for i in range(len(self.actorloss))], self.actorloss)
        plt.xlabel("reply times")
        plt.ylabel("actor loss")
        plt.show()

        plt.plot([i+1 for i in range(max_episodes)], ep_num)
        plt.xlabel("episode")
        plt.ylabel("ep_num")
        plt.show()




if __name__ == "__main__":
    env = Env_DQN()
    dqn_agent = DQN(env, time_steps=1)
    ep_opt, ep_min_objective = dqn_agent.train()
