

import numpy as np
import os
import time
import tensorflow as tf
import tensorlayer as tl

from util import QuantizeOutput

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

resume = True  # load model, resume from previous checkpoint?

# Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)  # (50000,)
print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)  # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))


def data_to_tfrecord(images, labels, filename):
    """Save data into TFRecord."""
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    # cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        # Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """Return tensor to read from TFRecord."""
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train ==True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])

        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)

        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)

        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

        # 5. Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)

        # 2. Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label

    

# Save data into TFRecord files
data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar10")

batch_size = 1
resume = True  # load model, resume from previous checkpoint?

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_test_, y_test_ = read_and_decode("test.cifar10", False)
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch(
        [x_test_, y_test_], batch_size=batch_size, capacity=50000, num_threads=32
    )

    def model(x_crop, y_, reuse):
        """For more simplified CNN APIs, check tensorlayer.org."""
        with tf.variable_scope("model", reuse=reuse):
            net = tl.layers.InputLayer(x_crop, name='input')
            output1 = tl.layers.Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')
            net = tl.layers.MaxPool2d(output1, (3, 3), (2, 2), padding='SAME', name='pool1')
            output2 = tl.layers.Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')
            net = tl.layers.MaxPool2d(output2, (3, 3), (2, 2), padding='SAME', name='pool2')
            net = tl.layers.FlattenLayer(net, name='flatten')
            output3 = tl.layers.DenseLayer(net, 384, act=tf.nn.relu, name='d1relu')
            output4 = tl.layers.DenseLayer(output3, 192, act=tf.nn.relu, name='d2relu')
            output5 = tl.layers.DenseLayer(output4, 10, act=None, name='output')

            return output1.outputs, output2.outputs, output3.outputs, output4.outputs, output5.outputs, output5

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        output1, output2, output3, output4, output5, network = model(x_test_batch, y_test_batch, False)

    sess.run(tf.global_variables_initializer())
    if resume:
        tl.files.load_and_assign_npz(sess=sess, name='./tensorRT_Quantize/model.npz', network=network)

    network.print_params(False)
    network.print_layers()

    #quantize the weight
    W = [0, 2, 4, 6, 8]
    #w1, w2, w3, w4 = sess.run([network.all_params[W[1]], network.all_params[W[2]], network.all_params[W[3]], network.all_params[W[4]]])
    M = sess.run([network.all_params[W[1]], network.all_params[W[2]], network.all_params[W[3]], network.all_params[W[4]]])

    qt = QuantizeOutput(128, 2048, 4)

    new_w = []
    scales = []

    for m in M:
        w_, scale_ = qt.weight_scale(m)
        new_w.append(w_)
        scales.append(scale_)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(int(len(y_test) / batch_size)):
        o1, o2, o3, o4 = sess.run([output1, output2, output3, output4])
        qt.find_max_output(o1, o2, o3, o4)

    for _ in range(int(len(y_test) / batch_size)):        
        o1, o2, o3, o4 = sess.run([output1, output2, output3, output4])
        qt.histogram(o1, o2, o3, o4)

    output_scale_ = qt.output_value_scale()


    cur_w = sess.run(network.all_params)
    
    final_w = []
    final_w.append(cur_w[0])
    final_w.append(cur_w[1])

    final_w.append(output_scale_[0])

    final_w.append(new_w[0])
    final_w.append(cur_w[3])
    final_w.append(scales[0] * output_scale_[0][0])

    final_w.append(output_scale_[1])

    final_w.append(new_w[1])
    final_w.append(cur_w[5])
    final_w.append(scales[1] * output_scale_[1][0])

    final_w.append(output_scale_[2])

    final_w.append(new_w[2])
    final_w.append(cur_w[7])
    final_w.append(scales[2] * output_scale_[2][0])

    final_w.append(output_scale_[3])

    final_w.append(new_w[3])
    final_w.append(cur_w[9])
    final_w.append(scales[3] *  output_scale_[3][0])

    np.savez('model1.npz', params=final_w)
    coord.request_stop()
    coord.join(threads)
    sess.close()



