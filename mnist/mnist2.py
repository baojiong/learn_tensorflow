#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 28 * 28
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率 - 指数衰减法
REGULARIZATION_RATE = 0.0001    # 正则化项在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率


def inference(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights",
                                  [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights",
                                  [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    return layer2


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    一个3层全链接网络前向传播过程，采用relu为激活函数。
    :param input_tensor:第一层的数据集
    :param avg_class:   滑动平均类
    :param weights1:    参数1
    :param biases1:     偏置项1
    :param weights2:    参数2
    :param biases2:     偏置项2
    :return:            无
    """

#   当没有提供滑动平均类时，直接使用参数的当前取值。
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    """
    mnist的训练过程。整个训练过程共涉及一下几个概念：
    1. 偏置项；2. 激活函数（已由inference函数封装）；3. 损失函数-交叉熵；4. 梯度下降； 5. 学习率； 6. 指数衰减; 7. 滑动平均模型
    :param mnist:   mnist数据集
    :return:        无
    """
# 第一步，前向传播，获得 y

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
    # 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # 偏置项个数是等于同层参数的矩阵的后一项，即下一层神经元的个数。
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

# 第二步，前向传播，获得带有移动平均模型的 average_y

    # 滑动平均模型，它可以使得模型在测试数据上更健壮，在使用随机梯度下降算法训练神经网络时，
    # 通过滑动平均模型可以在很多的应用中在一定程度上提高最终模型在测试数据上的表现。
    # 其实滑动平均模型，主要是通过控制衰减率来控制参数更新前后之间的差距，
    # 从而达到减缓参数的变化值（如，参数更新前是5，更新后的值是4，通过滑动平均模型之后，参数的值会在4到5之间），
    # 如果参数更新前后的值保持不变，通过滑动平均模型之后，参数的值仍然保持不变。
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # tf.trainable_variables 返回所有 当前计算图中所有通过tf.Variable(…)生成的变量，在获取变量时未标记 trainable=False 的变量集合
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

# 第三部，损失函数，loss
    '''
    tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，
    那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
    A = [[1,3,4,5,6]]  
    B = [[1,3,4], [2,4,1]]  
      
    with tf.Session() as sess:  
        print(sess.run(tf.argmax(A, 1)))
        print(sess.run(tf.argmax(B, 1)))
    
    输出：
    [4]
    [2 1]
    '''
    # 交叉熵, tf.argmax(y_, 1) -> y_向量中最大值的下标，即等于该数字的值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 正则化 - 防止过拟合
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    # 损失函数 = 平均交叉熵 + 正则化
    loss = cross_entropy_mean + regularization

# 第四步，学习率

    # 指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

# 第五步，反向传播，采用梯度下降算法

    # 实现梯度下降算法的优化器。(结合理论可以看到，这个构造函数需要的一个学习率就行了)
    # optimizer = GradientDescentOptimizer(learning_rate)
    # train_step = optimizer.minimize(loss, global_step=global_step)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 第六步，开始训练的准备工作
    '''
    在训练模型时我们每步训练可能要执行两种操作，op a, b
    这时我们就可以使用如下代码：

    with tf.control_dependencies([a, b]):
        c = tf.no_op(name='train')  # tf.no_op；什么也不做
    sess.run(c)
    '''
# 6.1 train_op 实际分2步：train_step （反向传播）和 variable_averages_op（滑动平均所有参数）
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

# 6.2 准确率公式，accuracy

    '''
    tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    A = [[1,3,4,5,6]]  
    B = [[1,3,4,3,2]]  
  
    with tf.Session() as sess:  
        print(sess.run(tf.equal(A, B)))  

    输出：
    [[ True  True  True False False]]
    '''
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # tf.cast(x, dtype, name=None) 将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 第七步，开始训练

    with tf.Session() as sess:

        # 7.1 初始化所有变量
        tf.global_variables_initializer().run()

        # 7.2 准备验证和测试数据，为输出作准备
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 7.3 输出验证集上的准确率
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))

            # 7.4 核心 - 运行训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 7.5 输出在测试集的准确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy "
              "using average model is %g " % (i, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("mnist_samples/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
