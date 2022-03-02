from datetime import datetime
import tensorflow as tf

# start_time = datetime.now()
#
# start_time = str(start_time)[:19]
# tran_tab = str.maketrans('- :', '___')
# plt_name = str(start_time).translate(tran_tab)
#
# print(plt_name, start_time)

# x = tf.constant([[[1, 2, 3, 4, 5, 6], [2, 2, 3, 4, 5, 6]], [[3, 2, 3, 4, 5, 6], [4, 2, 3, 4, 5, 6]]])
# print(x)
# temp = tf.reshape(x, [2, 2, 2, 3])
# print(temp)
# temp = tf.transpose(temp, perm=[0, 1, 3, 2])
# print(temp)
# temp = tf.reshape(temp, [2, 2, 6])
# print(temp)

# temp = tf.constant([[1, 2, 3, 4, 5, 6]])
# print(temp.shape.as_list())
# for _ in range(3):
#     temp = tf.reshape(temp, [-1, 3])
#     print(temp)
#     temp = tf.transpose(temp, perm=[1, 0])
#     print(temp)
#     temp = tf.reshape(temp, [6])
#     print(temp)
