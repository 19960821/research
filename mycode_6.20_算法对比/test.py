import numpy as np
import tensorflow as tf
# user_num = 10
# action_dim = user_num * 3
# data_size = np.random.uniform(2e5, 4e5, user_num)#--------数据量200-400K
# data_frequency = np.random.uniform(500, 1000, user_num)
# model_frequency = 300
# user_computing_ability = np.random.uniform(1e9, 2e9, user_num)
# model_size = 3e5
# bandwidth = 5e5
# power = 1
# user_location = np.random.uniform(0, 300, size=[user_num, 2])
# server_location = np.array([150,150])
# loss_exponent = -3
# noise_power = 1e-13
# server_computing_ability = 6e9
# energy_exponent = 1e-28#----------------------暂定此数
# time_tolerance = 1


# us_distance = np.array([])
# aa = np.square(user_location - server_location)
# for a in aa:
#     us_distance = np.append(us_distance, np.sqrt(a[0]+a[1]))

# user_bandwidth = bandwidth / user_num * np.ones(user_num)
# print("----------------user_bandwidth---------------\n", user_bandwidth)
# user_power = np.random.uniform(0, power, 8)
# user_power = np.append(user_power, 0)
# user_power = np.append(user_power, 0)
# print("----------------user_power---------------\n", user_power)
# user_ability = np.random.uniform(0, user_computing_ability, user_num)
# user_ability[9] = 0
# print("----------------user_ability---------------\n", user_ability)

# time_local_comp = (data_size * data_frequency) / user_ability
# print("----------------time_local_comp---------------\n", time_local_comp)
# print(type(time_local_comp[0]))
# #传输速率---一个数组
# rate = user_bandwidth * np.log(1 + (user_power * np.power(us_distance, loss_exponent)) / noise_power)
# print("----------------rate---------------\n", rate)
# #传输时延-----一个数组
# time_commu = model_size / rate
# print("----------------time_commu---------------\n", time_commu)
# print(type(time_commu[0]))
# #融合时延-----一个数
# time_aggregation = model_size * model_frequency / server_computing_ability
# print("----------------time_aggregation---------------\n", time_aggregation)

# # 当带宽、功率或能力为零时将值设置为零
# zero_indices = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0))

# print("---------------zero_indices---------------\n", zero_indices[0])
# zero_indices_int = tf.constant(zero_indices[0], dtype=tf.int32)
# print(zero_indices_int)
# # time_local_comp[zero_indices] = 0
# # print("----------------time_local_comp----new---------------\n", time_local_comp)
# # time_commu[zero_indices] = 0
# # print("----------------time_commu------new---------\n", time_commu)

# # #总时延------一个数
# # time_total = np.max(time_local_comp+time_commu) + time_aggregation*user_num
# # print("----------------time_total---------------\n", time_total)

# zero_indices = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0))
# zero_indices_int = tf.constant(zero_indices[0], dtype=tf.int32)
# new_values = tf.zeros(shape=(len(zero_indices[0]),), dtype=tf.float64)
# time_local_comp = tf.tensor_scatter_nd_add(time_local_comp, zero_indices_int, new_values)
# time_commu = tf.tensor_scatter_nd_add(time_commu, zero_indices_int, new_values)




# import tensorflow as tf

# # 创建一个 EagerTensor 对象
# tensor = tf.constant([1, 2, 3], dtype=tf.float32)

# # 要修改为0的索引位置
# indices_to_update = tf.constant([[0]], dtype=tf.int32)

# # 要更新的新值
# new_values = tf.constant([0.0], dtype=tf.float32)

# # 使用 tf.tensor_scatter_nd_update 更新指定索引位置的值为0
# updated_tensor = tf.tensor_scatter_nd_update(tensor, indices_to_update, new_values)

# print("原始张量:")
# print(tensor.numpy())
# print("\n更新后的张量:")
# print(updated_tensor.numpy())


# import h5py




# def print_h5py_object_info(obj, obj_name):
#     if isinstance(obj, h5py.Group):
#         print(f"Group: {obj_name}")
#         for key in obj.keys():
#             sub_obj = obj[key]
#             print_h5py_object_info(sub_obj, f"{obj_name}/{key}")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"Dataset: {obj_name} (Shape: {obj.shape}, Dtype: {obj.dtype})")
#     else:
#         print(f"Unknown object type: {obj_name}")

# # 打开HDF5文件
# file_path = r'ddpg_critic_final_episode5.h5'

# user_bandwidth = np.array([1,0,3,4,5])
# user_power = np.array([1,2,0,4,5])
# user_ability = np.array([1,2,3,0,5])

# aa = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0), 0, user_bandwidth)
# print(aa)

# # 当带宽、功率或能力为零时将值设置为零
# zero_indices = np.where((user_bandwidth == 0) | (user_power == 0) | (user_ability == 0))


# if len(zero_indices[0])>0:
#     new_values = tf.constant([0] * len(zero_indices[0]), dtype=tf.float64)
#     user_bandwidth = tf.tensor_scatter_nd_update(user_bandwidth, tf.expand_dims(zero_indices[0], axis=1), new_values)
#     user_power = tf.tensor_scatter_nd_update(user_power, tf.expand_dims(zero_indices[0], axis=1), new_values)
#     user_ability = tf.tensor_scatter_nd_update(user_ability, tf.expand_dims(zero_indices[0], axis=1), new_values)

# print(user_bandwidth)
# print(user_power)
# print(user_ability)



# def decimal_to_ternary_array(decimal_number, total_length):
#     # 将十进制数转换为三进制数的字符串表示
#     ternary_str = ''
#     if decimal_number == 0:
#         ternary_str = '0'
#     else:
#         while decimal_number > 0:
#             ternary_str = str(decimal_number % 3) + ternary_str
#             decimal_number //= 3
    
#     # 确保三进制数的总长度为 N，通过前置零填充
#     ternary_str = ternary_str.zfill(total_length)
    
#     # 将字符串转换为 NumPy 数组
#     ternary_array = np.array([int(char) for char in ternary_str], dtype=int)
    
#     return ternary_array

# # 示例：将十进制数 10 转换为长度为 5 的三进制数
# decimal_number = 11
# total_length = 5
# ternary_array = decimal_to_ternary_array(decimal_number, total_length)
# # print(ternary_array)
# # print("---------\n")
# print(ternary_array[1:2])

a = np.array([1,2,3])
a += np.array([4,5,6])
print(a)