# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tensorlayer as tl
import os
import numpy
import numpy as np
import librosa
from random import shuffle


class CNNConfig(object):
    """CNN配置参数"""
    learning_rate = 1e-2  # 学习率
    num_epochs = 1000  # 总迭代轮次
    batch_size = 200
    print_per_batch = 20
    save_tb_per_batch = 10



class ASRCNN(object):
    def __init__(self, input_x, input_y, config, width, height, num_classes, is_train=True, reuse=False):  # 20,80
        self.config = config
        # input_x = tf.reshape(self.input_x, [-1, height, width])
        input_x = tf.transpose(input_x, [0, 2, 1])
        self.input_x = tf.reshape(input_x, [-1, height, width, 1])
        self.input_y = input_y

        with tf.variable_scope("binarynet", reuse=reuse):
            net = tl.layers.InputLayer(self.input_x, name='input')
            net = tl.layers.Conv2d(net, 8, (10, 3), (10, 3), padding='SAME', b_init=None, name='bcnn0')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool0')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn0')

            net = tl.layers.Conv2d(net, 16, (3, 3), (1, 1), padding='SAME', b_init=None, name='bcnn1')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn1')

            net = tl.layers.Conv2d(net, 16, (2, 2), (2, 2), padding='VALID', b_init=None, name='bcnn2')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn2')

            net = tl.layers.Conv2d(net, 32, (3, 3), (1, 1), padding='VALID', b_init=None, name='bcnn3')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn3')

            net = tl.layers.FlattenLayer(net)
            # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop0')
            net = tl.layers.DenseLayer(net, n_units=num_classes, b_init=None, name='dense')
            net = tl.layers.BatchNormLayer(net, act=tf.identity, is_train=is_train, name='bn4')

            # 分类器
            self.logits = net.outputs
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.net = net


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return numpy.eye(num_classes)[labels_dense]

def read_files(files):
    labels = []
    features = []
    for ans, files in files.items():
        for file in files:
            wave, sr = librosa.load(file, mono=True)
            label = dense_to_one_hot(ans, 10)
            # label = [float(value) for value in label]
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr, n_mfcc=24)
            l = len(mfcc)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
            features.append(np.array(mfcc))
            # print('reading '+file)
    return np.array(features), np.array(labels)


def load_files(path='data/spoken_numbers_pcm/'):
    files = os.listdir(path)
    cls_files = {}
    for wav in files:
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        cls_files.setdefault(ans, [])
        cls_files[ans].append(path + wav)
    train_files = {}
    valid_files = {}
    test_files = {}
    for ans, file_list in cls_files.items():
        shuffle(file_list)
        all_len = len(file_list)
        train_len = int(all_len * 0.7)
        valid_len = int(all_len * 0.2)
        test_len = all_len - train_len - valid_len
        train_files[ans] = file_list[0:train_len]
        valid_files[ans] = file_list[train_len:train_len + valid_len]
        test_files[ans] = file_list[all_len - test_len:all_len]
    return train_files, valid_files, test_files


def batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(input_x, input_y, x_batch, y_batch):
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
    }
    return feed_dict


def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value


def train(argv=None):
    '''batch = mfcc_batch_generator()
    X, Y = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y  # overfit for now'''
    train_files, valid_files, test_files = load_files()
    train_features, train_labels = read_files(train_files)
    train_features = mean_normalize(train_features)
    print('read train files down')
    valid_features, valid_labels = read_files(valid_files)
    valid_features = mean_normalize(valid_features)
    print('read valid files down')
    test_features, test_labels = read_files(test_files)
    test_features = mean_normalize(test_features)
    print('read test files down')

    width = 24  # mfcc features
    height = 80  # (max) length of utterance
    classes = 10  # digits

    config = CNNConfig

    input_x = tf.placeholder(tf.float32, [None, width, height], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, classes], name='input_y')

    cnn_train = ASRCNN(input_x, input_y, config, width, height, classes, is_train=True, reuse=False)
    cnn_test = ASRCNN(input_x, input_y, config, width, height, classes, is_train=False, reuse=True)


    
    #session = tf.Session()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    cnn_train.net.print_params()
    cnn_train.net.print_layers()

    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn_train.loss)
    tf.summary.scalar("accuracy", cnn_train.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(config.num_epochs):
        # print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_features, train_labels)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            #神奇的feed_dict
            feed_dict = feed_data(input_x, input_y, x_batch, y_batch)
            session.run(cnn_train.optim, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn_train.loss, cnn_train.acc], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([cnn_test.loss, cnn_test.acc], feed_dict={input_x: valid_features,
                                                                                         input_y: valid_labels})
                print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={input_x: valid_features, input_y: valid_labels})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([cnn_test.loss, cnn_test.acc],
                                           feed_dict={input_x: test_features, input_y: test_labels})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))


if __name__ == '__main__':
    train()
    # test('data/spoken_numbers_pcm/9_Alex_260.wav')