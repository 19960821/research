# state     1.每个用户的带宽 4. 每个用户的计算资源 5. 每个用户的功率      user_num * 3
# action   1. 每个用户的带宽 2. 每个用户的计算能力  3. 每个用户的功率 
# reward    在step函数里面定义
# state_shape    ok
# action_dim    ok
# action_bound   ok
# action_shift   ok
# done      可以不要，然后修改DDPGtrain的流程 
# init()定义初始参数    
# reset()返回state
# step()返回next_state，reward，done----可以作为控制位控制流程？？

#-------此处可以定义一些不会变化的参数---------
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class Env_DQN():

    

    def __init__(self, ):
        self.user_num = 3
        self.action_dim = pow(3, self.user_num * 3)
        self.data_size = np.random.uniform(2e5, 4e5, self.user_num)#--------数据量200-400K
        self.data_frequency = np.random.uniform(500, 1000, self.user_num)
        self.model_frequency = 300
        self.user_computing_ability = np.random.uniform(1e9, 2e9, self.user_num)
        self.model_size = 3e5
        self.bandwidth = 5e6
        self.power = 1
        self.user_location = np.random.uniform(0, 300, size=[self.user_num, 2])
        self.server_location = np.array([150,150])
        self.loss_exponent = -3
        self.noise_power = 1e-13
        self.server_computing_ability = 6e9
        self.energy_exponent = 1e-28#----------------------暂定此数
        self.time_tolerance = 1

        self.ternary_len = self.user_num*3

        self.scaler = StandardScaler()

        self.b_step = (self.bandwidth / self.user_num) * 1e-3 
        self.a_step = 1e6
        self.p_step = 1e-3


        #-----计算距离------
        self.us_distance = np.array([])
        aa = np.square(self.user_location - self.server_location)
        for a in aa:
            self.us_distance = np.append(self.us_distance, np.sqrt(a[0]+a[1]))



        self.state_shape = (self.user_num*3,)   #可以是一个数组吗？ 如果是数组就是（user_num, 3）. 或者是一个一维向量 （user_num*3， ）


    def compute_time(self, user_ability, user_bandwidth, user_power):
        #本地计算时延-----一个数组
        time_local_comp = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0), 0, (self.data_size * self.data_frequency) / user_ability)
        # time_local_comp = (self.data_size * self.data_frequency) / user_ability
        #传输速率---一个数组
        rate = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0), 0, user_bandwidth * np.log(1 + (user_power * np.power(self.us_distance, self.loss_exponent)) / self.noise_power))
        # rate = user_bandwidth * np.log(1 + (user_power * np.power(self.us_distance, self.loss_exponent)) / self.noise_power)
        #传输时延-----一个数组
        time_commu = np.where(rate==0, 0, self.model_size / rate)
        # time_commu = self.model_size / rate
        #融合时延-----一个数--------0.015
        time_aggregation = self.model_size * self.model_frequency / self.server_computing_ability

        # # 当带宽、功率或能力为零时将值设置为零
        # zero_indices = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0))
        
        # if len(zero_indices[0])>0:
        #     new_values = tf.constant([0] * len(zero_indices[0]), dtype=tf.float64)
        #     time_local_comp = tf.tensor_scatter_nd_update(time_local_comp, tf.expand_dims(zero_indices[0], axis=1), new_values)
        #     rate = tf.tensor_scatter_nd_update(rate, tf.expand_dims(zero_indices[0], axis=1), new_values)
        #     time_commu = tf.tensor_scatter_nd_update(time_commu, tf.expand_dims(zero_indices[0], axis=1), new_values)

        #总时延------一个数----不会为零
        time_total = np.max(time_local_comp+time_commu) + time_aggregation*self.user_num

        # print("----------------time_local_comp------------", time_local_comp)
        # print("----------------time_commu------------", time_commu)
        return time_local_comp, rate, time_commu, time_aggregation, time_total
    

    def compute_energy(self, user_ability, user_power, time_commu):
        #本地计算能耗-------数组
        energy_local_comp = self.energy_exponent * self.data_size * self.data_frequency * np.square(user_ability)
        #通信能耗------数组
        energy_commu = user_power * time_commu
        #能耗之和----数
        energy_total = np.sum(energy_local_comp + energy_commu)
        return energy_local_comp, energy_commu, energy_total



    def reset(self):#返回一个state 
        # state     
        #  3. 每个用户的带宽 4. 每个用户的计算资源 5. 每个用户的功率
        user_bandwidth = self.bandwidth / self.user_num * np.ones(self.user_num)
        user_power = self.power - np.random.uniform(0, self.power, self.user_num)
        user_ability = self.user_computing_ability - np.random.uniform(0, self.user_computing_ability, self.user_num)

        # user_bandwidth_N = 2 * user_bandwidth / self.bandwidth - 1
        # user_ability_N = 2 * user_ability / self.user_computing_ability - 1
        # user_power_N = 2 * user_power / self.power - 1
        state_O = np.concatenate([user_bandwidth, user_ability, user_power])
        # print("---------原始state--------\n", state_O)
        state_N = self.scaler.fit_transform(state_O.reshape(-1, 1)).flatten()
        # print("---------归一化之后--------\n", state_N)
        return state_O, state_N
        


    def decimal_to_ternary_array(self, decimal_number, total_length):
        # 将十进制数转换为三进制数的字符串表示
        ternary_str = ''
        if decimal_number == 0:
            ternary_str = '0'
        else:
            while decimal_number > 0:
                ternary_str = str(decimal_number % 3) + ternary_str
                decimal_number //= 3
        
        # 确保三进制数的总长度为 N，通过前置零填充
        ternary_str = ternary_str.zfill(total_length)
        
        # 将字符串转换为 NumPy 数组
        ternary_array = np.array([int(char) for char in ternary_str], dtype=int)
        
        return ternary_array


    def step(self, cur_state, action):#next_state, reward, done, *_ = self.env.step(action)

        # print("---------action---------\n", action)
        # print("-------actionlen----------\n", len(action))

        # print("--------当前带宽--------\n", cur_state[:self.user_num])
        # print("--------当前资源--------\n", cur_state[self.user_num:self.user_num*2])
        # print("--------当前功率--------\n", cur_state[self.user_num*2:])

        action_temp = self.decimal_to_ternary_array(action, self.ternary_len)

        action_T = np.where(action_temp == 2, -1, action_temp)
        # print("-----------动作-------------\n", action_T)
        #0---不变，1----增大，2---减小
        done = False

        

        user_bandwidth = cur_state[:self.user_num] + action_T[:self.user_num] * self.b_step
        # print("---------user_bandwidth----------\n", user_bandwidth)
        b_index = np.where(user_bandwidth != 0, 1, 0)

        user_ability = cur_state[self.user_num:self.user_num*2] + action_T[self.user_num:self.user_num*2] * self.a_step
        # print("---------user_ability----------\n", user_ability)
        a_index = np.where(user_bandwidth != 0, 1, 0)

        user_power = cur_state[self.user_num*2:] + action_T[self.user_num*2:] * self.p_step
        # print("---------user_power----------\n", user_power)
        p_index = np.where(user_bandwidth != 0, 1, 0)


        f_num = np.sum(user_ability<=0)
        # print("-----------f----------\n", f_num)
        F_num = np.sum(user_ability>self.user_computing_ability)
        # print("-----------F----------\n", F_num)
        p_num = np.sum(user_power<=0)
        # print("-----------p----------\n", p_num)
        P_num = np.sum(user_power>self.power)
        # print("-----------P----------\n", P_num)
        b_num = np.sum(user_bandwidth<=0)

        if f_num + F_num + p_num + P_num + b_num> 0:
            done = True
            return done, 0, 0, 0, 0, 0, 0


                # state     1. 总时延 2. 所有用户的能耗之和 
        #  3. 所有用户的带宽之和 4. 每个用户的计算资源 5. 每个用户的功率
        time_local_comp, rate, time_commu, time_aggregation, time_total = self.compute_time(user_ability, user_bandwidth, user_power)
        energy_local_comp, energy_commu, energy_total = self.compute_energy(user_ability, user_power, time_commu)
        bandwidth = np.sum(user_bandwidth)
        # print("------------------bandwidth-----------\n", bandwidth)



        next_state_O = np.concatenate([user_bandwidth, user_ability, user_power])
        next_state_N = self.scaler.fit_transform(next_state_O.reshape(-1, 1)).flatten()
        
        r1 = time_total
        r11 = 1/time_total#-----ok
        # print("------------------r11-----------\n", r11)
        # print("------------------time_total-----------\n", r1)

        r2 = energy_total

        r22 = 0 if energy_total==0 else 1/energy_total#------ok
        # print("------------------r22-----------\n", r22)
        # print("------------------energy-----------\n", r2)

        
        # r33 = self.time_tolerance/time_total#-------ok
        # r33 = time_total/self.time_tolerance#-------ok
        r33 = time_total-self.time_tolerance
        # print("------------------r33-----------\n", r33)

       
        # r44 = 0 if bandwidth==0 else self.bandwidth/bandwidth#-----ok
        # r44 = bandwidth/self.bandwidth#-----ok/
        r44 = bandwidth-self.bandwidth
        # print("------------------r44-----------\n", r44)
 
        

        objective = r1 + r2
        # objective = r2

        flag = r33<=0 and r44<=0 #and r55==0 and r66==0 and r77==0 and r88==0#-------------错了。。。。

        available_num = sum(np.bitwise_and(np.bitwise_and(b_index, a_index), p_index))

        # reward = r11+r22

        # reward = r11 + r22 - 0.1*r33 - 1e-7*r44

        # reward = r11+r22 if flag else r11+r22-r33-r44

        reward = r11 + r22 - 0.1*r33 - 1e-7*r44
        # if r33>0:
        #     reward -= 0.1*r33
        # if r44>0:
        #     reward -= 1e-7*r44

        return done, next_state_O, next_state_N, reward, objective, flag, available_num
    

