import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "log"
BATCH_SIZE = 100
TRAIN_STEPS = 30000


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope('layer_name'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            pre_activate = tf.matmul(input_tensor, weights) + biases

            tf.summary.histogram(layer_name + '/pre_activations', pre_activate)

        activations = act(pre_activate, name='activation')

        tf.summary.histogram(layer_name + '/activations', activations)
        return activations

def main():
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
