# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# 给定一张图像， 随机调整图像的色彩。 因为调整亮度，对比度、饱和度和色相的顺序
# 会影响最后得到的结果，可以定义多种不同的顺序。 具体使用哪种顺序可以在训练数据
# 预处理时随机的选择一种。这样可以进一步降低无关因素对模型的影响。
def distort_color(image, color_ording=0):
    if color_ording == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ording == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # elif color_ording == 2:
    #    ...

    return tf.clip_by_value(image, 0.0, 1.0)


# 给定一张解码后的图像、目标图像的尺寸以及图像上的标注框， 此函数恶意对给出的图像进行预处理。
# 这个函数的输入图像是图像识别问题中原始的训练图像， 而输出则是神经网络模型的输入层。
# 注意这里只是处理模型
def preprocess_for_train(image, height, width, bbox):
    # 如果没有标注框，就认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体的大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图像调整成为神经网络输入层的大小。大小调整的算法是随机选择的。
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))
    # 随机右反转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))

    return distorted_image


image_raw_data = tf.gfile.FastGFile("photo.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # 运行6次获得6次不同的图像
    for i in range(6):
        # 将图像调整为 299x299
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()