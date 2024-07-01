import gym
import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from Prioritized_Replay import Memory
from env import *
from env_DQN import *
from DQN import *
import pandas as pd

# Original paper: https://arxiv.org/pdf/1509.02971.pdf
# DDPG with PER paper: https://cardwing.github.io/files/RL_course_report.pdf

tf.keras.backend.set_floatx('float64')


def actor(state_shape, action_dim, action_bound, action_shift, units=(400, 300)):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation='relu')(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)

    unscaled_output = Dense(action_dim, name="Out", activation='tanh')(x)
    scalar = action_bound * np.ones(action_dim)
    output = Lambda(lambda op: op * scalar)(unscaled_output)
    if np.sum(action_shift) != 0:
        output = Lambda(lambda op: op + action_shift)(output)  # for action range not centered at zero

    model = Model(inputs=state, outputs=output)

    return model, unscaled_output, scalar, output


def critic(state_shape, action_dim, units=(48, 24)):
    inputs = [Input(shape=state_shape), Input(shape=(action_dim,))]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L0", activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
    output = Dense(1, name="Out")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class NormalNoise:
    def __init__(self, mu, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class DDPG:
    def __init__(
            self,
            env,
            discrete=False,
            use_priority=False,
            lr_actor=1e-5,
            lr_critic=1e-3,
            # actor_units=(24, 16),
            actor_units=(400, 300),
            # critic_units=(24, 16),
            critic_units=(400, 300),
            noise='norm',
            sigma=0.15,
            tau=0.125,
            gamma=0.85,
            batch_size=64,
            memory_cap=100000
    ):
        self.env = env
        self.state_shape = env.state_shape  # shape of observations
        self.action_dim = env.action_dim  # number of actions
        self.discrete = discrete
        self.action_bound = env.action_bound
        self.action_shift = env.action_shift
        # self.deta = 1e-9*self.action_bound
        # self.deta = (1-np.random.rand())*self.action_bound#------可以设置成可行解的边界值，讲道理不是可行解应该也可以
        self.use_priority = use_priority


        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)
        if noise == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma)
        else:
            self.noise = NormalNoise(mu=np.zeros(self.action_dim), sigma=sigma)

        # Define and initialize Actor network
        self.actor, self.unscaled_output, self.scalar, self.output = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)    
        # self.actor = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
        self.actor_target, *_ = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.)

        # Define and initialize Critic network
        self.critic = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_target = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic, self.critic_target, tau=1.)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # Tensorboard
        self.summaries = {}
        self.actorloss = []
        self.get_unscaled_output = tf.keras.backend.function([self.actor.input], [self.unscaled_output])
        self.get_output = tf.keras.backend.function([self.actor.input], [self.output])

    def act(self, state, add_noise=True):
        # print("state:---------", state)
        state = np.expand_dims(state, axis=0).astype(np.float32)
        # print(state)
        
        unscaled_output_val = self.get_unscaled_output([state])[0]
        output_val = self.get_output([state])[0]
        scalar_val = self.scalar
        # Print values
        # print("Unscaled Output:", unscaled_output_val)
        # print("Scalar:", scalar_val)
        # print("Scaled Output:", output_val)


        a = self.actor.predict(state)
        # print("--------------网络预测----------\n", a[0])
        a += self.noise() * add_noise * self.action_bound
        # print("--------------加噪声后----------\n", a[0])
        # a = tf.clip_by_value(a, -self.action_bound + self.action_shift + self.deta, self.action_bound + self.action_shift)
        # a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)
        # print("--------------约束范围后----------\n", a[0])

        self.summaries['q_val'] = self.critic.predict([state, a])[0][0]

        return a

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
        print(self.critic.summary())

    def remember(self, state, action, reward, next_state):
        if self.use_priority:
            action = np.squeeze(action)
            transition = np.hstack([state, action, reward, next_state])
            self.memory.store(transition)
        else:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            self.memory.append([state, action, reward, next_state])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        if self.use_priority:
            tree_idx, samples, ISWeights = self.memory.sample(self.batch_size)
            split_shape = np.cumsum([self.state_shape[0], self.action_dim, 1, self.state_shape[0]])
            states, actions, rewards, next_states = np.hsplit(samples, split_shape)
        else:
            ISWeights = 1.0
            samples = random.sample(self.memory, self.batch_size)
            # print("-----------sample---------", samples[0])
            s = np.array(samples, dtype=object).T
            states, actions, rewards, next_states = [np.vstack(s[i, :]).astype(float) for i in range(4)]

        next_actions = self.actor_target.predict(next_states)
        q_future = self.critic_target.predict([next_states, next_actions])
        target_qs = rewards + q_future * self.gamma

        # train critic
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions])
            td_error = q_values - target_qs
            critic_loss = tf.reduce_mean(ISWeights * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)  # compute critic gradient
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # update priority
        if self.use_priority:
            abs_errors = tf.reduce_sum(tf.abs(td_error), axis=1)
            self.memory.batch_update(tree_idx, abs_errors)

        # # train actor with action penalty
        # with tf.GradientTape() as tape:
        #     actions_pred = self.actor(states)
        #     actor_critic_loss = -tf.reduce_mean(self.critic([states, actions_pred]))

        #     # Define a penalty for actions close to 0
        #     # Adjust the scale of the penalty as needed
        #     action_penalty_scale = 1e-9
        #     action_penalty = action_penalty_scale * tf.reduce_mean(tf.square(actions_pred))

        #     actor_loss = actor_critic_loss + action_penalty  # Include penalty in actor loss
        #     self.actorloss.append(actor_loss)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))
            self.actorloss.append(actor_loss)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # tensorboard info
        self.summaries['critic_loss'] = critic_loss
        self.summaries['actor_loss'] = actor_loss

    


    def train(self, max_episodes=50, max_steps=50):#50 10

        ep_reward = []
        ep_opt = []
        ep_opt_R = []
        ep_opt_A = []
        ep_min_objective = []
        ep_min_objective_R = []
        ep_min_objective_A = []
        ep_fin_objective = []
        ep_action = []
        ep_action_R = []
        ep_action_A = []
        # ep_available = []
        ep_num = []

        episode, steps, total_reward = 0, 0, 0
        cur_state, user_bandwidth, user_power, user_ability = self.env.reset()
        # print("cur_state's type is: \n", type(cur_state))
        # print("--------------cur_state:------------\n", cur_state)

        opt = -10
        opt_R = -10
        opt_A = -10
        min_objective = 0.6
        min_objective_R = 0.6
        min_objective_A = 0.6
        opt_action = np.ones(3*self.env.user_num)
        opt_action_R = np.ones(3*self.env.user_num)
        opt_action_A = np.ones(3*self.env.user_num)
        # opt_flag = False
        
        while episode<max_episodes:
            opt_num = 0
            opt_num_R = 0
            opt_num_A = 0
            while steps<max_steps:
                # print("------------------cur_state-----------\n", cur_state)
                a = self.act(cur_state)  # model determine action given state
                action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space


                # -----------------随机--------------
                UB = user_bandwidth - np.random.random(user_bandwidth.shape) * user_bandwidth
                UP = user_power - np.random.random(user_power.shape) * user_power
                UA = user_ability - np.random.random(user_ability.shape) * user_ability
              
                # UB = (1-np.random.rand())*user_bandwidth
                # UP = (1-np.random.rand())*user_power
                # UA = (1-np.random.rand())*user_ability
                action_R = np.concatenate([UB, UA, UP])
                # -----------------平均--------------
                UB_A = 0.8*user_bandwidth
                UP_A = 0.8*user_power
                UA_A = 0.8*user_ability
                action_A = np.concatenate([UB_A, UA_A, UP_A])
                # -----------------DQN--------------
                # action_DQN = 


                done, next_state, reward, objective, flag, available_num = self.env.step(action)  # perform action on env
                # ---------------随机------------
                done_R, next_state_R, reward_R, objective_R, flag_R, available_num_R = self.env.step(action_R)

                done_A, next_state_A, reward_A, objective_A, flag_A, available_num_A = self.env.step(action_A)


                if done:
                    break

                # print("------------------reward-----------", reward)
                self.remember(cur_state, a, reward, next_state)  # add to memory
                self.replay()  # train models through memory replay
                update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
                update_target_weights(self.critic, self.critic_target, tau=self.tau)
                cur_state = next_state
                total_reward += reward
                steps += 1
                if flag and available_num==self.env.user_num:# and available_num>=0.5*env.user_num
                    if reward>opt:
                        opt = reward
                    if objective<min_objective:
                        min_objective=objective
                        opt_action = action
                        opt_num = available_num
                
                # ------------随机-----------
                if flag_R and available_num_R==self.env.user_num:# and available_num>=0.5*env.user_num
                    if reward_R>opt_R:
                        opt_R = reward_R
                    if objective_R<min_objective_R:
                        min_objective_R=objective_R
                        opt_action_R = action_R
                        opt_num_R = available_num_R


                if flag_A and available_num_A==self.env.user_num:# and available_num>=0.5*env.user_num
                    if reward_A>opt_A:
                        opt_A = reward_A
                    if objective_A<min_objective_A:
                        min_objective_A=objective_A
                        opt_action_A = action_A
                        opt_num_A = available_num_A

            print("episode {}: {} total reward, {} steps".format(
                    episode, total_reward, steps))
            # self.save_model("ddpg_actor_episode{}.h5".format(episode),
            #                         "ddpg_critic_episode{}.h5".format(episode))
            self.noise.reset()
            ep_reward.append(total_reward)
            ep_fin_objective.append(objective)
            ep_opt.append(opt)
            # # --------随机-----------
            ep_opt_R.append(opt_R)
            ep_opt_A.append(opt_A)
            ep_min_objective.append(min_objective)
            # --------随机-----------
            ep_min_objective_R.append(min_objective_R)
            ep_min_objective_A.append(min_objective_A)


            ep_action.append(opt_action)
            ep_action_R.append(opt_action_R)
            ep_action_A.append(opt_action_A)
            # ep_available.append(opt_flag)
            ep_num.append(opt_num)
            
            cur_state, user_bandwidth, user_power, user_ability = self.env.reset()
            steps = 0
            total_reward = 0
            episode += 1
        # self.save_model("ddpg_actor_final_episode{}.h5".format(episode),
        #                 "ddpg_critic_final_episode{}.h5".format(episode))
        
        # plot the reward
        # fig_reward = plt.figure()
        
        return ep_opt, ep_opt_R, ep_opt_A, ep_min_objective, ep_min_objective_R, ep_min_objective_A




if __name__ == "__main__":
    
    episodes=50
    steps=50
    average_num = 5


    # ep_opt = np.zeros(episodes)
    # ep_opt_R = np.zeros(episodes)
    # ep_opt_A = np.zeros(episodes)
    # ep_opt_D = np.zeros(episodes)
    ep_min_objective = np.zeros(episodes)
    ep_min_objective_R = np.zeros(episodes)
    ep_min_objective_A = np.zeros(episodes)
    ep_min_objective_D = np.zeros(episodes)


    for i in range(average_num):


        env_DQN = Env_DQN()
        dqn_agent = DQN(env_DQN, time_steps=1)
        ep_opt_D_temp, ep_min_objective_D_temp = dqn_agent.train(max_episodes=episodes, max_steps=steps)

    
        env = Env()
        ddpg = DDPG(env)
        ep_opt_temp, ep_opt_R_temp, ep_opt_A_temp, ep_min_objective_temp, ep_min_objective_R_temp, ep_min_objective_A_temp = ddpg.train(max_episodes=episodes, max_steps=steps)

        # ep_opt += ep_opt_temp
        # ep_opt_R += ep_opt_R_temp
        # ep_opt_A += ep_opt_A_temp
        # ep_opt_D += ep_opt_D_temp
        ep_min_objective += ep_min_objective_temp
        ep_min_objective_R += ep_min_objective_R_temp
        ep_min_objective_A += ep_min_objective_A_temp
        ep_min_objective_D += ep_min_objective_D_temp
    
    
    data = {
    "Episode": [i + 1 for i in range(episodes)],
    "DDPG": ep_min_objective / average_num,
    "RAM": ep_min_objective_R / average_num,
    "FAM": ep_min_objective_A / average_num,
    "DQN": ep_min_objective_D / average_num,

    }
    df = pd.DataFrame(data)
    file_name = f"objective_data_average_num_{average_num}.csv"

    df.to_csv(file_name, index=False)




    # plt.plot([i+1 for i in range(episodes)], ep_opt/average_num, label = "DDPG")
    # plt.plot([i+1 for i in range(episodes)], ep_opt_R/average_num, label = "RAM")
    # plt.plot([i+1 for i in range(episodes)], ep_opt_A/average_num, label = "FAM")
    # plt.plot([i+1 for i in range(episodes)], ep_opt_D/average_num, label = "DQN")
    # plt.xlabel("episode")
    # plt.ylabel("reward")
    # plt.legend()
    # plt.show()
    

    # plt.plot([i+1 for i in range(episodes)], ep_min_objective/average_num, label = "DDPG")
    # plt.plot([i+1 for i in range(episodes)], ep_min_objective_R/average_num, label = "RAM")
    # plt.plot([i+1 for i in range(episodes)], ep_min_objective_A/average_num, label = "FAM")
    # plt.plot([i+1 for i in range(episodes)], ep_min_objective_D/average_num, label = "DQN")
    # plt.xlabel("Episode")
    # plt.ylabel("Objective")
    # plt.legend()
    # plt.show()

    plt.plot(df["Episode"], df["DDPG"], label="DDPG")
    plt.plot(df["Episode"], df["RAM"], label="RAM")
    plt.plot(df["Episode"], df["FAM"], label="FAM")
    plt.plot(df["Episode"], df["DQN"], label="DQN")
    plt.xlabel("Episode")
    plt.ylabel("Objective")
    plt.legend()
    plt.show()

