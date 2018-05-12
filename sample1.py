import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(2, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
#sess.run(w1.initializer)
#sess.run(w2.initializer)
init_op = tf.global_variables_initializer()
sess.run(init_op)

#print(sess.run(y))
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.3, 0.6]]}))

cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y, le-10, 1.0)))

learning_rate = 0.001

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
sess.run(train_step)
