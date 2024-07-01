# state     1. 总时延 2. 所有用户的能耗之和 
# action   1. 每个用户的计算能力 
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


class Env():

    
    def __init__(self, i):
        self.user_num = i
        self.action_dim = self.user_num
        self.data_size = np.random.uniform(2e5, 4e5, self.user_num)#--------数据量200-400K
        self.data_frequency = np.random.uniform(500, 1000, self.user_num)
        self.model_frequency = 300
        self.user_computing_ability = np.random.uniform(1e9, 2e9, self.user_num)
        self.model_size = 3e5

        self.bandwidth = (5e6 / self.user_num) * np.ones(self.user_num)

        self.power = np.ones(self.user_num)

        self.user_location = np.random.uniform(0, 300, size=[self.user_num, 2])
        self.server_location = np.array([150,150])
        self.loss_exponent = -3
        self.noise_power = 1e-13
        self.server_computing_ability = 6e9
        self.energy_exponent = 1e-28#----------------------暂定此数
        self.time_tolerance = 1

        #-----计算距离------
        self.us_distance = np.array([])
        aa = np.square(self.user_location - self.server_location)
        for a in aa:
            self.us_distance = np.append(self.us_distance, np.sqrt(a[0]+a[1]))


        self.action_bound = self.user_computing_ability / 2 * np.ones(self.user_num)
        self.action_shift = self.user_computing_ability / 2 * np.ones(self.user_num)

        self.state_shape = (2,)   #可以是一个数组吗？ 如果是数组就是（user_num, 3）. 或者是一个一维向量 （user_num*3， ）


    def compute_time(self, user_ability):
        #本地计算时延-----一个数组
        time_local_comp = (self.data_size * self.data_frequency) / user_ability
        #传输速率---一个数组
        rate = self.bandwidth * np.log(1 + (self.power * np.power(self.us_distance, self.loss_exponent)) / self.noise_power)
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
    

    def compute_energy(self, user_ability, time_commu):
        #本地计算能耗-------数组
        energy_local_comp = self.energy_exponent * self.data_size * self.data_frequency * np.square(user_ability)
        #通信能耗------数组
        energy_commu = self.power * time_commu
        #能耗之和----数
        energy_total = np.sum(energy_local_comp + energy_commu)
        return energy_local_comp, energy_commu, energy_total



    def reset(self):#返回一个state 类型是ndarray--------------此时还没做对应化处理-------还有除数为零的风险需要处理
        # state      1. 总时延 2. 所有用户的能耗之和 
        user_ability = self.user_computing_ability - np.random.uniform(0, self.user_computing_ability, self.user_num)
        
        time_local_comp, rate, time_commu, time_aggregation, time_total = self.compute_time(user_ability)
        
        energy_local_comp, energy_commu, energy_total = self.compute_energy(user_ability, time_commu)
        
        # state = np.concatenate([[time_total], [energy_total], [bandwidth], user_ability, user_power])
        # data_scaled = 2 * (data - min_val) / (max_val - min_val) - 1
        time_total_N = 2 * time_total / self.time_tolerance - 1
        state = np.concatenate([[time_total_N], [energy_total]])
        return state
        





    def step(self, action):#next_state, reward, done, *_ = self.env.step(action)

        done = False

        f_num = np.sum(action<=0)
        F_num = np.sum(action>self.user_computing_ability)

        if f_num + F_num  > 0:#--------没限制bandwidth的范围----下限
            done = True
            return done, 0, 0, 0, 0


        time_local_comp, rate, time_commu, time_aggregation, time_total = self.compute_time(action)
        energy_local_comp, energy_commu, energy_total = self.compute_energy(action, time_commu)


        #------归一化--------
        time_total_N = 2 * time_total / self.time_tolerance - 1

        next_state = np.concatenate([[time_total_N], [energy_total]])
        
        r1 = time_total
        r11 = 1/time_total#-----ok

        r2 = energy_total

        r22 = 0 if energy_total==0 else 1/energy_total#------ok

        
        r33 = time_total-self.time_tolerance

 
        objective = r1 + r2
        # objective = r2

        flag = r33<=0

        reward = r11 + r22 - r33

        return done, next_state, reward, objective, flag
    

