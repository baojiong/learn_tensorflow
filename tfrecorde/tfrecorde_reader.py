import tensorflow as tf

reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer('output.tfrecords')

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    print('%d' % i)
    image, label, pixel = sess.run([images, labels, pixels])
