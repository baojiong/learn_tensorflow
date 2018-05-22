#coding=utf-8

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def recognize_a_number():
    image_raw_data = tf.gfile.FastGFile("4.jpg", 'rb').read()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
        img_data = tf.image.resize_images(img_data, [28, 28], method=0)
        img_data = sess.run(tf.image.rgb_to_grayscale(img_data))
        print(img_data.shape)

        reshaped = tf.reshape(img_data[:, :, 0], [1, 784])
        #print(reshaped.get_shape())
        #print(reshaped.eval())

        #plt.imshow(img_data)
        # plt.show()

        return predict(reshaped)


def predict(image_dataset):

    input_image_tensor = tf.convert_to_tensor(image_dataset, dtype=tf.float32)

    reshaped = tf.reshape(input_image_tensor, [1, 784])

    y = mnist_inference.inference(reshaped, None)

    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(y)
            y_ = tf.argmax(y, 1)
            print(y_.eval())
            return y_.eval()


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    recognize_a_number()


if __name__ == '__main__':
    tf.app.run()

