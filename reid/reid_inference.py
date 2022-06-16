import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path

ROOTPATH = Path(__file__).parent.parent
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', f'{ROOTPATH}/reid/logs/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', f'{ROOTPATH}/reid/data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'test', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')


class Reid:
    IMAGE_W = 60
    IMAGE_H = 160

    def __init__(self):
        tf.reset_default_graph()
        FLAGS.batch_size = 1

        tf.placeholder(tf.float32, name='learning_rate')
        self.images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3], name='images')
        labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        tf.Variable(0, name='global_step', trainable=False)
        weight_decay = 0.0005
        images1, images2 = self._preprocess(self.images, self.is_train)
        logits = self._network(images1, images2, weight_decay)
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        self.inference = tf.nn.softmax(logits)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("REID Error: Bad checkpoints path!")

    def compare(self, image_1, image_2):
        image1 = cv2.resize(image_1, (Reid.IMAGE_W, Reid.IMAGE_H))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = np.reshape(image1, (1, Reid.IMAGE_H, Reid.IMAGE_W, 3)).astype(float)
        image2 = cv2.resize(image_2, (Reid.IMAGE_W, Reid.IMAGE_H))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = np.reshape(image2, (1, Reid.IMAGE_H, Reid.IMAGE_W, 3)).astype(float)
        test_images = np.array([image1, image2])

        feed_dict = {self.images: test_images, self.is_train: False}
        prediction = self.sess.run(self.inference, feed_dict=feed_dict)
        return bool(not np.argmax(prediction[0]))

    @staticmethod
    def _preprocess(images, is_train):
        def train():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3])
                split[i] = tf.image.resize_images(split[i], [Reid.IMAGE_H + 8, Reid.IMAGE_W + 3])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [Reid.IMAGE_H + 8, Reid.IMAGE_W + 3, 3])
                    split[i][j] = tf.random_crop(split[i][j], [Reid.IMAGE_H, Reid.IMAGE_W, 3])
                    split[i][j] = tf.image.random_flip_left_right(split[i][j])
                    split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                    split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                    split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3]),
                    tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3])]

        def val():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3])
                split[i] = tf.image.resize_images(split[i], [Reid.IMAGE_H, Reid.IMAGE_W])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [Reid.IMAGE_H, Reid.IMAGE_W, 3])
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3]),
                    tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, Reid.IMAGE_H, Reid.IMAGE_W, 3])]

        return tf.cond(is_train, train, val)

    @staticmethod
    def _network(images1, images2, weight_decay):
        with tf.variable_scope('network'):
            # Tied Convolution
            conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       name='conv1_1')
            pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
            conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       name='conv1_2')
            pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')

            conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       name='conv2_1')
            pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
            conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       name='conv2_2')
            pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

            # Cross-Input Neighborhood Differences
            trans = tf.transpose(pool1_2, [0, 3, 1, 2])
            shape = trans.get_shape().as_list()
            m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
            reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
            f = tf.multiply(reshape, m1s)

            trans = tf.transpose(pool2_2, [0, 3, 1, 2])
            reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
            g = []
            pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
            for i in range(shape[2]):
                for j in range(shape[3]):
                    g.append(pad[:, :, :, i:i + 5, j:j + 5])

            concat = tf.concat(g, axis=0)
            reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
            g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
            reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
            reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
            k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
            k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

            # Patch Summary Features
            l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
            l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

            # Across-Patch Features
            m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
            pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
            m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
            pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

            # Higher-Order Relationships
            concat = tf.concat([pool_m1, pool_m2], axis=3)
            reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
            fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 2, name='fc2')

            return fc2


if __name__ == "__main__":
    reid = Reid()
    for i in [0, 1]:
        im1 = cv2.imread(f"{ROOTPATH}/test_data/{i}_1.jpg")
        im2 = cv2.imread(f"{ROOTPATH}/test_data/{i}_2.jpg")
        res = reid.compare(im1, im2)
        print(res)
