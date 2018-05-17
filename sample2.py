#coding=utf-8
'''
训练神经网络的全部过程：
1、定义神经网络的结构和前向传播的输出结果。
2、定义损失函数以及选择反向传播优化算法。
3、生成会话，并在训练集上反复运行反向传播优化算法。
无论神经网络怎么变，这3步是不变的。
'''
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#损失函数：H(p, q) = Sigma p(x)log q(x)，其中 p -> y_：标签值；q - > y：预测值
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
#反向传播
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#随机种子
rdm = RandomState(1)
dataset_size = 128
#np.random.rand: a convenience function for np.random.uniform(0, 1)
#从 0-1中取 128行2列的数据集
'''
X = [[6.79068837e-01 9.18601778e-01]
 [4.02024891e-04 9.76759149e-01]
 [3.76580315e-01 9.73783538e-01]
 ...
 [6.04716101e-01 8.28845808e-01]]

Y = [[0], [1], [1],... [1]] - 打标
'''
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

#训练5000次，数据集已固定，每次从数据集中取出1个batch_size的样例。
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entrop = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d trainning step(s), cross entropy on all data is %g" %(i, total_cross_entrop))

    print(sess.run(w1))
    print(sess.run(w2))

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularzer(lambda1)(var))

    return var



