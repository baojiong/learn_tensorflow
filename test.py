import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = a + b

# with tf.Session() as sess:
#     sess.run(result)
#     print(result.eval())

sess = tf.Session()
with sess.as_default():
    print(result.eval())

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
biases = tf.Variable(tf.zeros([3]))

w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initial_value() * 2)
