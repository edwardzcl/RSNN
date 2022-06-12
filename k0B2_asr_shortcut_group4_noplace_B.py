# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tensorlayer as tl
import os
import numpy
import numpy as np
import librosa
from random import shuffle

import ast
print(os.getcwd())
tf.reset_default_graph()

import argparse

parser = argparse.ArgumentParser()
# quantization level
parser.add_argument('--k', type=int, default=0)
# upper bound
parser.add_argument('--B', type=int, default=2)
# learning rate
parser.add_argument('--learning_rate', type=float, default=0.01)
# resume from previous checkpoint
parser.add_argument('--resume', type=ast.literal_eval, default=False)
# training or inference
parser.add_argument('--mode', type=str, default='training')
args = parser.parse_args()

print(args.k, args.B, args.learning_rate, args.resume, args.mode)


class CNNConfig(object):
    """CNN配置参数"""
    #结构体
    #learning_rate = 1e-2  # 学习率
    learning_rate = args.learning_rate  # 学习率
    num_epochs = 1000  # 总迭代轮次
    batch_size = 200
    print_per_batch = 20
    save_tb_per_batch = 10
    print_freq = 10
    k = args.k
    B = args.B



class ASRCNN(object):
    def __init__(self, input_x, input_y, config, width, height, num_classes, is_train=True, reuse=False):  # 20,80
        self.config = config
        # input_x = tf.reshape(self.input_x, [-1, height, width])
        input_x = tf.transpose(input_x, [0, 2, 1])
        self.input_x = tf.reshape(input_x, [-1, height, width, 1])
        self.input_y = input_y

        with tf.variable_scope("binarynet", reuse=reuse):
            net_in = tl.layers.InputLayer(self.input_x, name='input')
            net0 = tl.layers.Conv2d(net_in, 16, (10, 3), (10, 3), padding='SAME', b_init=None, name='bcnn0')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool0')
            net0 = tl.layers.BatchNormLayer(net0, act=tf.nn.relu, is_train=is_train, name='bn0')
            net0 = tl.layers.Quant_Layer(net0, config.k, config.B)

            #这下子成了标准shorucut，在芯片上的shortcut是elementwisr形式的add，可以灵活处理，充分利用，规则化，简单化，尽管输入输出可能会有些没有充分利用
            #可以考虑一下均衡分配
            #考虑一下非方形，小型5*5或者其他mapping算子
            #通盘考虑，采用特定步长或者填充方式的价值，带来的卷积核的实际意义
            #调节系数，匹配运算能力，提升精度，不规则卷积带来的挑战
            shortcut0 = tl.layers.Quant_Conv2d(net0, 64, (2, 2), (2, 2), padding='SAME', b_init=None, name='shortcut0')
            shortcut0 = tl.layers.BatchNormLayer(shortcut0, act=tf.nn.relu, is_train=is_train, name='bn_shortcut0')
            shortcut0 = tl.layers.Quant_Layer(shortcut0, config.k, config.B)
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            #shortcut0 = tl.layers.BatchNormLayer(shortcut0, act=tf.nn.relu, is_train=is_train, name='bn_shortcut0')

            #需要考虑下是否使用input_frame的tick还是层间的tick_relative，此外残差网络需要考虑正泄露以及重叠，权值组合，分支的复制问题
            #考虑一下叠加的可行性
            net1 = tl.layers.Quant_Conv2d(net0, 32, (3, 3), (1, 1), padding='SAME', b_init=None, name='bcnn1')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            net1 = tl.layers.BatchNormLayer(net1, act=tf.nn.relu, is_train=is_train, name='bn1')
            net1 = tl.layers.Quant_Layer(net1, config.k, config.B)

            #保证参数量还是相等的
            net2 = tl.layers.Quant_Conv2d(net1, 64, (2, 2), (2, 2), padding='VALID', b_init=None, name='bcnn2')
            #64已经是单group的极限
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
            #net2 = tl.layers.BatchNormLayer(net2, act=tf.nn.relu, is_train=is_train, name='bn2')
            #net2 = tl.layers.Quant_Layer(net2, config.k, config.B)

            #方便了core上叠加，方便了中值量化
            #注意core.py里面的layer基类，list_remove_repeat方法，当有多分支时，顺序不完全按照网络构建顺序，而是要去重
            #需要好好规划多分支顺序，以保证snn运行正确，要求一条分支写到底，而不是并行式交替书写 
            net2 = tl.layers.BatchNormLayer(net2, act=tf.nn.relu, is_train=is_train, name='bn_shortcut1')
            net2 = tl.layers.Quant_Layer(net2, config.k, config.B)
            shortcut0 = tl.layers.ElementwiseLayer([shortcut0, net2], combine_fn=tf.add, act=None, name='elementwise0')

            #还可以制造更多group以及只放部分层
            #net3= tl.layers.Quant_Conv2d(shortcut0, 64, (3, 3), (1, 1), padding='VALID', b_init=None, name='bcnn3')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
            #net3 = tl.layers.BatchNormLayer(net3, act=tf.nn.relu, is_train=is_train, name='bn3')
            #net3 = tl.layers.Quant_Layer(net3, config.k, config.B)

            #最可怕的是，这里也需要进行group
            #这里如果不把全连接层mapping在芯片上，可以实现每个输出group都拥有64个通道
            net3=[]
            for i in range(4):
                #不使用128，是为了生成帧的时候，排除检查报错
                net3.append(tl.layers.Quant_Conv2d(shortcut0[:,:,:,i::4], 64, (3, 3), (1, 1), padding='VALID', b_init=None, name='bcnn3_'+str(i+1)))
                net3[i] = tl.layers.BatchNormLayer(net3[i], act=tf.nn.relu, is_train=is_train, name='bn3_'+str(i+1))
                net3[i] = tl.layers.Quant_Layer(net3[i], config.k, config.B)
                #net3[i].outputs = (net3[i].outputs+1)/2

            net3 = tl.layers.ConcatLayer(layers = [net3[i] for i in range(4)], name ='concat_layer1')

            #考虑最后一层要不要放在芯片上，group与shortcut联合带来的挑战，二脉冲与四脉冲的权衡
            net3 = tl.layers.FlattenLayer(net3)
            # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop0')
            #还是想要把这一层放上去啊
            net4 = tl.layers.DenseLayer(net3, n_units=num_classes, b_init=None, name='dense')
            #在上位机上做bn，也是没有价值的，简单为要
            #net4 = tl.layers.BatchNormLayer(net4, act=tf.identity, is_train=is_train, name='bn4')
            #这个激活函数很重要
            #对与全连接层，bn也是分通道的，而不是只有一套参数，除非是ln
            #要么也将其融入量化层（权值或者阈值），要么在训练时直接取消这一层的bn

            # 分类器
            self.logits = net4.outputs
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.net = net4


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
        train_len = int(all_len * 0.8)
        valid_len = int(all_len * 0.1)
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


def train(args=None):
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


    model_file_name = tf.train.latest_checkpoint('./cnn_model/')
    #globe, global_step可能是个问题

    if args.resume:
        print("Load existing model " + "!" * 10)
        saver.restore(session, model_file_name)

    if args.mode == 'training':

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

            if (epoch + 1) % (config.print_freq) == 0:
                print("Save npz model " + "!" * 10)
                #saver = tf.train.Saver()
                #save_path = saver.save(sess, model_file_name)
                # you can also save model into npz
                tl.files.save_npz(cnn_train.net.all_params, name='model_kws.npz', sess=session)
                # and restore it as follow:
                # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    test_loss, test_accuracy = session.run([cnn_test.loss, cnn_test.acc],
                                           feed_dict={input_x: test_features, input_y: test_labels})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))


if __name__ == '__main__':
    train(args)
    # test('data/spoken_numbers_pcm/9_Alex_260.wav')