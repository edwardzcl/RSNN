#python的import真的很强大
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tensorlayer as tl
import os
import numpy
import numpy as np
import scipy.io as io
import pdb
import librosa
from random import shuffle

# import spiking function
from spiking_ulils import label_encoder
from spiking_ulils import Conv2d, Conv2d_add, BatchNorm2d, BatchNorm2d_shortcut, Relu
from spiking_ulils import Flatten
from spiking_ulils import Linear


import ast
print(os.getcwd())
tf.reset_default_graph()

import argparse
parser = argparse.ArgumentParser()
# quantization level
parser.add_argument('--k', type=int, default=0)
# upper bound
parser.add_argument('--B', type=int, default=2)
# add noise
parser.add_argument('--noise_ratio', type=float, default=0)

# learning rate
parser.add_argument('--learning_rate', type=float, default=0.01)
# resume from previous checkpoint
parser.add_argument('--resume', type=ast.literal_eval, default=True)
# training or inference，对于SNN推理，默认为testing模式
parser.add_argument('--mode', type=str, default='testing')

args = parser.parse_args()

print(args.k, args.B, args.noise_ratio, args.learning_rate, args.resume, args.mode)

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

            #需要考虑下是否使用input_frame的tick还是层间的tick_relative，此外残差网络需要考虑正泄露以及重叠，权值组合，分支的复制问题
            #考虑一下叠加的可行性
            net1 = tl.layers.Quant_Conv2d(net0, 32, (3, 3), (1, 1), padding='SAME', b_init=None, name='bcnn1')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            net1 = tl.layers.BatchNormLayer(net1, act=tf.nn.relu, is_train=is_train, name='bn1')
            net1 = tl.layers.Quant_Layer(net1, config.k, config.B)

            #这下子成了标准shorucut，在芯片上的shortcut是elementwisr形式的add，可以灵活处理，充分利用，规则化，简单化，尽管输入输出可能会有些没有充分利用
            #可以考虑一下均衡分配
            #考虑一下非方形，小型5*5或者其他mapping算子
            #通盘考虑，采用特定步长或者填充方式的价值，带来的卷积核的实际意义
            #调节系数，匹配运算能力，提升精度，不规则卷积带来的挑战
            shortcut0 = tl.layers.Quant_Conv2d(net0, 64, (2, 2), (2, 2), padding='SAME', b_init=None, name='shortcut0')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            #shortcut0 = tl.layers.BatchNormLayer(shortcut0, act=tf.nn.relu, is_train=is_train, name='bn_shortcut0')

            net2 = tl.layers.Quant_Conv2d(net1, 64, (2, 2), (2, 2), padding='VALID', b_init=None, name='bcnn2')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
            #net2 = tl.layers.BatchNormLayer(net2, act=tf.nn.relu, is_train=is_train, name='bn2')
            #net2 = tl.layers.Quant_Layer(net2, config.k, config.B)

            #方便了core上叠加，方便了中值量化
            shortcut0 = tl.layers.ElementwiseLayer([shortcut0, net2], combine_fn=tf.add, act=None, name='elementwise0')
            shortcut0 = tl.layers.BatchNormLayer(shortcut0, act=tf.nn.relu, is_train=is_train, name='bn_shortcut0')
            shortcut0 = tl.layers.Quant_Layer(shortcut0, config.k, config.B)

            #还可以制造更多group以及只放部分层
            #net3= tl.layers.Quant_Conv2d(shortcut0, 64, (3, 3), (1, 1), padding='VALID', b_init=None, name='bcnn3')
            #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
            #net3 = tl.layers.BatchNormLayer(net3, act=tf.nn.relu, is_train=is_train, name='bn3')
            #net3 = tl.layers.Quant_Layer(net3, config.k, config.B)

            #最可怕的是，这里也需要进行group
            #这里如果不把全连接层mapping在芯片上，可以实现每个输出group都拥有64个通道
            net3=[]
            for i in range(4):
                net3.append(tl.layers.Quant_Conv2d(shortcut0[:,:,:,i::4], 64, (3, 3), (1, 1), padding='VALID', b_init=None, name='bcnn3_'+str(i+1)))
                net3[i] = tl.layers.BatchNormLayer(net3[i], act=tf.nn.relu, is_train=is_train, name='bn3_'+str(i+1))
                net3[i] = tl.layers.Quant_Layer(net3[i], config.k, config.B)
                #net3[i].outputs = (net3[i].outputs+1)/2

            net3 = tl.layers.ConcatLayer(layers = [net3[i] for i in range(4)], name ='concat_layer1')

            #考虑最后一层要不要放在芯片上，group与shortcut联合带来的挑战，二脉冲与四脉冲的权衡
            net3 = tl.layers.FlattenLayer(net3)
            # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop0')
            net4 = tl.layers.DenseLayer(net3, n_units=num_classes, b_init=None, name='dense')
            #直接查看膜电位，键位设置很灵活
            self.debug = net4.outputs
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
            self.snn_in = net0
            #self.debug = net4.outputs
            


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return numpy.eye(num_classes)[labels_dense]

def read_files(files):
    #这里没有进行多类别shuffle，但进行了字典到列表，以及numpy数组（尽管有其他类型）的转换
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
    #python文件读取并不是按照原始排列顺序，有它自己的规则，主要针对文件名
    files = os.listdir(path)
    #采用了字典，且按类别，这个语音数据集的组织以及标签方式还是很规整的
    cls_files = {}
    for wav in files:
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        cls_files.setdefault(ans, [])
        cls_files[ans].append(path + wav)
    #采用了字典
    train_files = {}
    valid_files = {}
    test_files = {}
    for ans, file_list in cls_files.items():
        #这里采用了shuffle的方式进行数据集处理，但是每次resume都会重新执行一次，因此最好不要每次都进行随机的划分并继续训练，可以存为一个固定的文件，或者不shuffle
        #所以再训练精度提升是有问题的，好在测试情况，直接用来测试了，但是似乎测试也需要进行数据集划分，也需要shuffle，这是一个需要解决的问题，但其实可以进行单次点到点访问
        #数据集划分比例，以及数据集选取，数据样本，过拟合，处理方式都可以进一步研究，数据集小了。随机性还是很大
        #python应该不与matlab一样具有很强的输入输出可重构
        #python字符串是真的字符串，不论中英文，不涉及字节串，尽管真实存储空间不一样
        #这里imagenet数据集处理有很相似的地方，按类别shuffle，还是很不一样的方式
        #python，tensorflow很强的计算图后端运行机制
        shuffle(file_list)
        all_len = len(file_list)
        #修改数据集配置比例
        train_len = int(all_len * 0.8)
        valid_len = int(all_len * 0.1)
        test_len = all_len - train_len - valid_len
        train_files[ans] = file_list[0:train_len]
        valid_files[ans] = file_list[train_len:train_len + valid_len]
        test_files[ans] = file_list[all_len - test_len:all_len]
        #后期肯定需要进行多类别shuffle
    return train_files, valid_files, test_files


def batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1
    #果然进行了多类别shuffle
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


def test_ann(session, cnn_train, cnn_test, input_x, input_y, train_features, train_labels, valid_features, valid_labels, test_features, test_labels, saver, args, config):
    '''batch = mfcc_batch_generator()
    X, Y = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y  # overfit for now'''

    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'
 
    #这个就必须先创建目录
    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn_train.loss)
    tf.summary.scalar("accuracy", cnn_train.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)


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

    test_loss, test_accuracy, debug = session.run([cnn_test.loss, cnn_test.acc, cnn_test.debug],
                                           feed_dict={input_x: test_features, input_y: test_labels})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))
    print(debug.shape)
    #debug = debug.transpose(0, 3, 1, 2)
    print(debug)


#ann函数写完，再写snn

class MyNet():
    def __init__(self):
        #步长受限，mapping的padding受限，但卷积核大小不受限制
        #可以考虑移花接木的手段，但是中间层可能需要来回转，mapping的时候也无法mapping非规则的，但可以mapping指定层，这里直接使用第二层
        self.conv1 = Conv2d(in_channels=16, n_filter=32, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B, noise_ratio=args.noise_ratio)   
        self.bn1 = BatchNorm2d(n_channel=32, momentum=0.1)
        self.relu1 = Relu()
        
        #仿真工具也需要对残差网络进行支持，对group的支持在时空间mapping以及2.0芯片的animal实验中得到支持
        #实例化的顺序，其实也无所谓了，毕竟不像标准框架里面进行记录
        self.shortcut_conv1 = Conv2d(in_channels=16, n_filter=64, filter_size=(2, 2), padding=0, stride=2, bn=0, k=args.k, B=args.B)
        #self.shortcut_bn1 = BatchNorm2d(n_channel=64, momentum=0.1)
        #self.shortcut_relu1 = Relu()

        #连续层，跨接
        self.conv2 = Conv2d(in_channels=32, n_filter=64, filter_size=(2, 2), padding=0, stride=2, bn=0, k=args.k, B=args.B)

        #作为一个单独类型层
        self.conv_add1 = Conv2d_add(n_channel=64, element_wise=1, bn=1, k=args.k, B=args.B)

        #排列顺序很重要，起标识作用，与shortcut类型层以及产生的conv2d_add层配合使用
        self.bn2 = BatchNorm2d_shortcut(n_channel=64, momentum=0.1)
        self.relu2 = Relu()

        #这里要支持element_wise的通道相加，可以认为有一个断层，需要在forward函数中加以支持

        #concat层，C4
        self.conv3=[]
        self.bn3=[]
        self.relu3=[]
        for i in range(4):
            self.conv3.append(Conv2d(in_channels=16, n_filter=64, filter_size=(3, 3), padding=0, stride=1, k=args.k, B=args.B))
            self.bn3.append(BatchNorm2d(n_channel=64, momentum=0.1))
            self.relu3.append(Relu()) 


        #self.conv3 = Conv2d(in_channels=64, n_filter=64, filter_size=(3, 3), padding=0, stride=1, k=args.k, B=args.B)
        #self.bn3 = BatchNorm2d(n_channel=64, momentum=0.1)
        #self.relu3 = Relu()

        self.flatten = Flatten()
        
        # ȫ���Ӳ�
        self.fc1 = Linear(dim_in=1024, dim_out=10, use_ternary=False)
        #全连接层之后这里不要加bn，实在不行可以使用1*1卷积融合bn

        #引入了所有参数，尽管有些参数是用来转化的，有些参数在SNN推理的时候不使用
        #这个参数其实也很重要，但是没有用到，就无所谓了
        #self.parameters = self.conv1.params + self.bn1.params + self.shortcut_conv1.params + \
        #                    self.conv2.params + self.conv_add1.params + self.bn2.params + self.conv3.params + self.bn3.params + \
        #                    self.fc1.params


        #可能有更好的办法处理列表
        self.parameters_list = [self.conv1.params + self.bn1.params + \
        self.shortcut_conv1.params + \
        self.conv2.params + \
        self.conv_add1.params + self.bn2.params] + \
        [self.conv3[i].params + self.bn3[i].params for i in range(4)] + \
        [self.fc1.params]


        self.parameters = []
        for i in range(len(self.parameters_list)):
            self.parameters = self.parameters + self.parameters_list[i]
        
        #可能有更简捷的办法书写列表
        self.dummy_layers = []
        dummy_layers = [self.conv3]
        dummy_layers_bn = [self.bn3]
        for i in range(1):
            for j in range(len(dummy_layers[i])):
                self.dummy_layers.append(dummy_layers[i][j])
                self.dummy_layers.append(dummy_layers_bn[i][j])

        self.dummy_layers = [self.shortcut_conv1, \
        self.conv1, self.bn1, \
        self.conv2, \
        self.conv_add1, self.bn2] + \
        self.dummy_layers + \
        [self.fc1]   

        #关键是这个参数，直接关系到convert_assign_params函数，因此一定要与npz文件排列顺序一致，要求也是一条分支写到底，而不是并行式交替书写
        #self.dummy_layers = [self.shortcut_conv1, self.conv1, self.bn1, self.conv2, self.conv_add1, self.bn2, self.conv3, self.bn3, self.fc1]                                        
    
    def __call__(self, X, t, mode='train'):
        """
        mode: ����ѵ���׶λ��ǲ��Խ׶�. train ���� test
        """
        return self.forward(X, t, mode)
    # spiking network inference during multiple time steps
    def forward(self, X, t, mode):
        # the first layer is usually a pixel-to-spike encoding layer

        #实例化中的Conv2d模块的count在这里的多分支架构，与tensorlayer的参数排列顺序一样，采用一维展开
        #多分支架构，在这里的forward函数里，不需要特别注意，这里只管数据流，但风格最好一致
        io.savemat('./' + '/' + 'spiking_input_feature_map', {'spiking_input_feature_map': X.transpose(2, 3, 1, 0)})
        shortcut_conv1_out, shortcut_conv1_spike_num, shortcut_conv1_sop_num = self.shortcut_conv1(X, t)

        #采用套接的方式，这里使用多分支
        conv1_out, conv1_spike_num, conv1_sop_num = self.conv1(X, t)
        
        conv2_out, conv2_spike_num, conv2_sop_num = self.conv2(conv1_out, t)

        #残差的时间展开，可能更加有挑战，甚至对于流水化的SNN
        #这一层就比较特殊，没有时间信息，脉冲计算在软件仿真以及芯片上不一样
        #膜电位通道累积
        add_conv2_out, add_conv2_spike_num, add_conv2_sop_num = self.conv_add1(shortcut_conv1_out, conv2_out, t)

        conv3_out=[]
        conv3_spike_num=0
        conv3_sop_num=0
        for i in range(4):
            out, spike, sop = self.conv3[i](add_conv2_out[:,i::4,:,:], t)
            conv3_out.append(out)
            conv3_spike_num += spike
            conv3_sop_num += sop
        
        conv3_out = np.concatenate(conv3_out, 1)    

        io.savemat('./' + '/' + 'spiking_output_feature_map', {'spiking_output_feature_map': conv3_out.transpose(2, 3, 1, 0)})

        #conv3_out, conv3_spike_num, conv3_sop_num = self.conv3(add_conv2_out, t)
        
        # the last layer output the membrane potential value indexing category   
        flat_out = self.flatten(conv3_out, t)
        
        fc1_out = self.fc1(flat_out, t)

        # spike number
        #spike的位置
        spike_num = conv1_spike_num + add_conv2_spike_num + conv3_spike_num
        # synaptic operations
        #这里的第一层也需要放在芯片上
        #sop的位置,这里没有加add_conv2_sop_num，因为可以忽略
        sop_num = conv1_sop_num + shortcut_conv1_sop_num + conv2_sop_num + conv3_sop_num
         
        return fc1_out, spike_num, sop_num, fc1_out
    

    def convert_assign_params(self, params, quant_level, upper_bound):
        tag = 0
        converted_params = []
        
        for index, layer in enumerate(self.dummy_layers):
            
            if layer.type == 'conv':         
               #self.layers[index].params[0] = params[tag].transpose(3, 2, 0, 1)
               # in this paper, we didn't quantize the weights, use_ternary is always false
               #self.dummy_layers[index].params[0][:,:,:,:] = self._ternary_operation(params[tag].transpose(3, 2, 0, 1))
               self.dummy_layers[index].params[0][:,:,:,:] = params[tag].transpose(3, 2, 0, 1)
               converted_params.append(params[tag])
               tag = tag + 1
            elif layer.type == 'conv_add':
                # BN layers need to be scaled
                #参考animal中的分层取整操作，这里采用套接的方式，可以直接取整
                #这里的灵活度有限，需要紧接bn层，直接获取下一层bn参数
                #这一层没有参数
                #需要注意层的排列顺序
                tag = tag + 0
            elif layer.type == 'bn':
                # BN layers need to be scaled
                #参考animal中的分层取整操作，这里采用套接的方式，可以直接取整
                for i in range((2**quant_level)*upper_bound):
                   self.dummy_layers[index-1].params[2][i][:] = np.ceil((1 / 2**(quant_level+1) + i / (2**quant_level) - params[tag]) * (2**quant_level * np.sqrt(params[tag+3] + 1e-5)) / params[tag+1] + \
                   2**quant_level * params[tag+2])
                   #仿真统一使用空间展开模式
                   #适用于空间展开模式
                   converted_params.append(self.dummy_layers[index-1].params[2][i][:])
                   converted_params.append(np.zeros_like(self.dummy_layers[index-1].params[2][i][:]))
                """
                #适用于时间展开模式
                converted_params.append(self.dummy_layers[index-1].params[2][0][:])  
                #对于单脉冲，将两者设置为相等
                if (2**quant_level)*upper_bound == 1:
                    bound_single_spike = self.dummy_layers[index-1].params[2][0][:]
                    converted_params.append(bound_single_spike - self.dummy_layers[index-1].params[2][0][:])
                elif (2**quant_level)*upper_bound > 1:
                    converted_params.append(self.dummy_layers[index-1].params[2][1][:] - self.dummy_layers[index-1].params[2][0][:])
                """
                tag = tag + 4
            elif layer.type == 'bn_shortcut':
                # BN layers need to be scaled
                #参考animal中的分层取整操作，这里采用套接的方式，可以直接取整
                #这里的灵活度有限，需要紧接bn层，直接获取下一层bn参数，配合conv_add层使用
                for i in range((2**quant_level)*upper_bound):
                    #注意conv2d_add没有参数，权值参数，它的上一层就是conv2d_add
                    #在matlab帧生成器中，自动贴合上层主干卷积参数
                   self.dummy_layers[index-1].params[0][i][:] = np.ceil((1 / 2**(quant_level+1) + i / (2**quant_level) - params[tag]) * (2**quant_level * np.sqrt(params[tag+3] + 1e-5)) / params[tag+1] + \
                   2**quant_level * params[tag+2])
                   #仿真统一使用空间展开模式
                   #适用于空间展开模式
                   converted_params.append(self.dummy_layers[index-1].params[0][i][:])
                   converted_params.append(np.zeros_like(self.dummy_layers[index-1].params[0][i][:]))
                """
                #适用于时间展开模式
                converted_params.append(self.dummy_layers[index-1].params[0][0][:])  
                #对于单脉冲，将两者设置为相等
                if (2**quant_level)*upper_bound == 1:
                    bound_single_spike = self.dummy_layers[index-1].params[0][0][:]
                    converted_params.append(bound_single_spike - self.dummy_layers[index-1].params[0][0][:])
                elif (2**quant_level)*upper_bound > 1:
                    converted_params.append(self.dummy_layers[index-1].params[0][1][:] - self.dummy_layers[index-1].params[0][0][:])
                """
                tag = tag + 4
            elif layer.type == 'fc':
                # just like the convolutional layer
                self.dummy_layers[index].params[0][:,:] = params[tag]
                converted_params.append(self.dummy_layers[index].params[0][:,:])
                tag = tag + 1

        converted_params = np.array(converted_params, dtype=object)
        io.savemat('./' + '/' + 'converted_params', {'converted_params': converted_params})



def test_snn(session, cnn_test, input_x, input_y, test_datas, quant_level, upper_bound, test_labels, network, n_data, batch_size, time_step):
    """
    function: snn test function entrance, test_labels need use one hot encoding
    return: generate four log files: spike_num.txt, sop_num, accuracy.txt and final SNN accuracy on test set
    """
    f1 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/spike_num.txt', 'w')
    f2 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/sop_num.txt', 'w')
    f3 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/accuracy.txt', 'w')

    #batch_iter只针对于train_data
    #图像以及标签自动会进行维度转换
    #这里的列表也很讲究
    test_datas = session.run(cnn_test.snn_in.outputs,
                                           feed_dict={input_x: test_datas, input_y: test_labels})
    test_datas = test_datas.transpose(0, 3, 1, 2)
    #框架多用float32，numpy默认采用float64，bn也有误差，还有其他误差
    test_labels = np.array(test_labels, np.float64)                                       
    #test_labels = label_encoder(test_labels, 10)
    n_data = test_labels.shape[0]
    n_correct = 0
    for i in range(0, n_data, batch_size):
        batch_datas = test_datas[i : i + batch_size] * 2**quant_level
        batch_labels = test_labels[i : i + batch_size]
        for t in range(time_step):
            if t == 0:
                net_out, spike_num, sop_num, debug = network(batch_datas, t, mode='test')
                #令人比较惊讶的是，置信度很高，基本上最后一层输出膜电位，只有一个目标是正的
                print(debug.shape)
                print(debug)
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(sop_num) + '\n')
                f3.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
            else:
                net_out, spike_num, sop_num = network(np.zeros_like(batch_datas), t, mode='test')
                #令人比较惊讶的是，置信度很高，基本上最后一层输出膜电位，只有一个目标是正的
                print(debug.shape)
                print(debug)
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(sop_num) + '\n')
                f3.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
        n_correct += np.sum(predict == np.argmax(batch_labels, axis=1))
        print('-----------------------Batch_number: ', i / batch_size, ' completed-----------------------')
        print(np.sum(predict == np.argmax(batch_labels, axis=1)) / batch_size)
        
    test_acc = n_correct / n_data
    f1.close()
    f2.close()
    f3.close()
    #print(conv3_out[0,0,:,0])
    return test_acc

#同名
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

model_file_name = tf.train.latest_checkpoint('./cnn_model/')
#globe, global_step可能是个问题

if args.resume:
    print("Load existing model " + "!" * 10)
    saver.restore(session, model_file_name)

#test_ann(session, cnn_train, cnn_test, input_x, input_y, train_features, train_labels, valid_features, valid_labels, test_features[0:1,:,:], test_labels[0:1,:], saver, args, config)
test_ann(session, cnn_train, cnn_test, input_x, input_y, train_features, train_labels, valid_features, valid_labels, test_features[:,:,:], test_labels[:,:], saver, args, config)


# define SNN instance
mynet = MyNet()

# load parameter
model = np.load('model_kws.npz')
params = model['params']
#这里需要截取一段，采用套接
params = params[5:]

mynet.convert_assign_params(params, args.k, args.B)

# total time steps
time_step = 1

#其实可以在时间以及空间全部一次性，仿真，或者测试算法
#test_acc = test_snn(session, cnn_test, input_x, input_y, test_features[0:1,:,:], args.k, args.B, test_labels[0:1,:], network=mynet, n_data=240, batch_size=1, time_step=time_step)
test_acc = test_snn(session, cnn_test, input_x, input_y, test_features[:,:,:], args.k, args.B, test_labels[:,:], network=mynet, n_data=240, batch_size=10, time_step=time_step)

print(test_acc)
