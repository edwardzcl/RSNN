import numpy as np

def label_encoder(label, num_class):
    """
    function: on-hot encoding
    label: 0，1，2, ..., num_class-1
    num_class: number of class
    return: one-hot vector of m samples, shape=(m, num_class)
    """
    tmp = np.eye(num_class)
    return tmp[label]
    

class Conv2d():
    count = 0
    def __init__(self, in_channels, n_filter, filter_size, padding, stride, bn=1, k=1, B=2, noise_ratio=0):
        """
        function: spiking convolution with positive and negative dynamics
        parameters:
            in_channel: number of input channles
            n_filter: number of filter
            filter_size: filter size (h_filter, w_filter)
            padding: padding size
            stride: convolution stride
        """
        Conv2d.count += 1
        self.in_channels = in_channels
        self.n_filter = n_filter
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
        self.thresholds = np.zeros([(2**k)*B, n_filter])
        self.leakages = np.zeros([n_filter])
        self.bn=bn
        self.precisions = [k, B]
        self.noise_ratio = noise_ratio
        self.layer_num = Conv2d.count
        self.debug = 1
        
        # initialize convolution parameters W, b
        self.W = np.random.randn(n_filter, self.in_channels, self.h_filter, self.w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((n_filter, 1))
        
        self.params = [self.W, self.b, self.thresholds, self.leakages]
        self.type = 'conv'
        
    def __call__(self, X, time_step):
        # compute for output feature size
        self.n_x, _, self.h_x, self.w_x = X.shape
        self.h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        self.w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

        # initialize neuron states at the first time step
        if time_step == 0:
            self.spike_num = 0
            self.sop_num = 0
            self.mem = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])

        self.spike_out = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])

        
        # instance of Img2colIndices
        self.img2col_indices = Img2colIndices((self.h_filter, self.w_filter), self.padding, self.stride)
        
        return self.forward(X, time_step)
    
    def forward(self, X, time_step):
        # img2col for input feature
        self.x_col = self.img2col_indices.img2col(X)
        
        # reshape W
        self.w_row = self.W.reshape(self.n_filter, -1)
        
        # foward using matrix multiplication
        out = self.w_row @ self.x_col + self.b  
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_x)
        out = out.transpose(3, 0, 1, 2)
        
        # membrane potential accumulation
        self.mem = self.mem + out

        if self.bn == 1:
            # leak
            self.mem = self.mem + np.repeat(np.repeat(np.repeat(self.leakages[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0)
            # generate spikes (scatter-and-gather)   
            if self.noise_ratio > 0:
               for i in range((2**self.precisions[0])*self.precisions[1]):
                   mask = np.where(np.random.rand(self.mem.shape[0], self.mem.shape[1], self.mem.shape[2], self.mem.shape[3]) > self.noise_ratio, 1, 0) 
                   self.spike_out = self.spike_out + np.where(self.mem >= np.repeat(np.repeat(np.repeat(self.thresholds[i,:][:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                   self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), 1, 0) * mask
            else:
                for i in range((2**self.precisions[0])*self.precisions[1]):
                   self.spike_out = self.spike_out + np.where(self.mem >= np.repeat(np.repeat(np.repeat(self.thresholds[i,:][:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                   self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), 1, 0)

            # count spikes
            self.spike_num = self.spike_num + np.sum(self.spike_out)
            # count synaptic operations for snn
            self.sop_num = self.sop_num + np.sum(self.x_col) * self.w_row.shape[0] * 2 * (2**self.precisions[0]) * self.precisions[1]
            #self.mem_trace[t] = self.mem_trace + np.abs(self.x_col)
            self.neurons = self.spike_out.shape[0] * self.spike_out.shape[1] * self.spike_out.shape[2] * self.spike_out.shape[3] * (2**self.precisions[0]) * self.precisions[1]
            
            #只产生膜电位，而不输出脉冲的情况，不应该打印
            print('Time_step: ', time_step, 'Layer: ', self.layer_num, ', spiking rate: ', np.sum(self.spike_out)/self.neurons)
        else:
            #这里的无bn跟神经网络里面的没有bn训练不是一回事，而是代表这里直接输出膜电位
            #无论有没有bn，主要的sop都在卷积这里，不在bn也不在其它层，如conv2d_add，尤其是在脉冲计算领域（与多位宽实数值计算的优缺点，有待比较）
            # count spikes
            self.spike_num = self.spike_num + 0
            # count synaptic operations for snn
            self.sop_num = self.sop_num + np.sum(self.x_col) * self.w_row.shape[0] * 2 * (2**self.precisions[0]) * self.precisions[1]
            #self.mem_trace[t] = self.mem_trace + np.abs(self.x_col)
            self.neurons = self.spike_out.shape[0] * self.spike_out.shape[1] * self.spike_out.shape[2] * self.spike_out.shape[3] * (2**self.precisions[0]) * self.precisions[1]
            self.spike_out = self.mem
        
        #这个参数暂时只作为参数层计数作用
        self.debug = self.debug + 1
        #对bn=0，这个脉冲数以及sop数没有意义
        return self.spike_out, self.spike_num, self.sop_num

    
class Img2colIndices():
    """
    a simple img2col implementation
    """
    def __init__(self, filter_size, padding, stride):
        """
        parameters:
            filter_shape: filter size (h_filter, w_filter)
            padding: padding size
            stride: convolution stride
        """
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
    
    def get_img2col_indices(self, h_out, w_out):
        """
        parameters:
            h_out: height of output feature
            w_out: width of output feature
        return:
            k: shape=(filter_height*filter_width*C, 1), index for channle
            i: shape=(filter_height*filter_width*C, out_height*out_width), index for row
            j: shape=(filter_height*filter_width*C, out_height*out_width), index for col
        """
        i0 = np.repeat(np.arange(self.h_filter), self.w_filter)
        i1 = np.repeat(np.arange(h_out), w_out) * self.stride
        i = i0.reshape(-1, 1) + i1
        i = np.tile(i, [self.c_x, 1])
        
        j0 = np.tile(np.arange(self.w_filter), self.h_filter)
        j1 = np.tile(np.arange(w_out), h_out) * self.stride
        j = j0.reshape(-1, 1) + j1
        j = np.tile(j, [self.c_x, 1])
        
        k = np.repeat(np.arange(self.c_x), self.h_filter * self.w_filter).reshape(-1, 1)
        
        return k, i, j
    
    def img2col(self, X):
        """
        parameters:
            x: input feature map，shape=(batch_size, channels, height, width)
        return:
            function of img2col, shape=(h_filter * w_filter*chanels, batch_size * h_out * w_out)
        """
        self.n_x, self.c_x, self.h_x, self.w_x = X.shape

        # compute for output feature size
        h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception("Invalid dimention")
        else:
            h_out, w_out = int(h_out), int(w_out)
        
        # 0 padding
        x_padded = None
        if self.padding > 0:
            x_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = X
        
        # call img2col_indices
        self.img2col_indices = self.get_img2col_indices(h_out, w_out)
        k, i, j = self.img2col_indices
        
        # transpose
        cols = x_padded[:, k, i, j]  # shape=(batch_size, h_filter*w_filter*n_channel, h_out*w_out)
        cols = cols.transpose(1, 2, 0).reshape(self.h_filter * self.w_filter * self.c_x, -1)  # reshape
        
        return cols
    
    def col2img(self, cols):
        """
        col2img: inverse process for img2col
        parameters:
            cols: shape=(h_filter*w_filter*n_chanels, batch_size*h_out*w_out)
        """
        # reshape
        cols = cols.reshape(self.h_filter * self.w_filter * self.c_x, -1, self.n_x)
        cols = cols.transpose(2, 0, 1)
        
        h_padded, w_padded = self.h_x + 2 * self.padding, self.w_x + 2 * self.padding
        x_padded = np.zeros((self.n_x, self.c_x, h_padded, w_padded))
        
        k, i, j = self.img2col_indices
        
        np.add.at(x_padded, (slice(None), k, i, j), cols)
        
        if self.padding == 0:
            return x_padded
        else:
            return x_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]


class Conv2d_add():
    count = 0
    def __init__(self, n_channel, element_wise=1, bn=1, k=1, B=2, noise_ratio=0):
        """
        function: spiking convolution with positive and negative dynamics
        parameters:
            in_channel: number of input channles
            n_filter: number of filter
            filter_size: filter size (h_filter, w_filter)
            padding: padding size
            stride: convolution stride
        """
        Conv2d_add.count += 1
        self.channel = n_channel
        self.thresholds = np.zeros([(2**k)*B, n_channel])
        self.leakages = np.zeros([n_channel])
        self.bn=bn
        self.element_wise = element_wise
        self.precisions = [k, B]
        self.noise_ratio = noise_ratio
        self.layer_num = Conv2d.count
        self.debug = 10

        # initialize convolution parameters W, b
        #self.W = np.random.randn(n_filter, self.in_channels, self.h_filter, self.w_filter) / np.sqrt(n_filter / 2.)
        #self.b = np.zeros((n_filter, 1))
        
        self.params = [self.thresholds, self.leakages]
        
        #未来可以增加concat等其他功能
        self.type = 'conv_add'
        
    def __call__(self, X1, X2, time_step):
        # compute for output feature size
        #size unchanged
        self.n_x1, self.c_x1, self.h_x1, self.w_x1 = X1.shape
        self.n_x2, self.c_x2, self.h_x2, self.w_x2 = X2.shape
        #self.h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        #self.w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        self.n_x = self.n_x1
        self.c_x = self.c_x1
        self.h_out = self.h_x1
        self.w_out = self.w_x1
        if self.n_x1 != self.n_x2 or self.c_x1 != self.c_x2 or self.h_x1 != self.h_x2 or self.w_x1 != self.w_x2:
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

        # initialize neuron states at the first time step
        if time_step == 0:
            self.spike_num = 0
            self.sop_num = 0
            self.mem = np.zeros([self.n_x, self.channel, self.h_out, self.w_out])

        self.spike_out = np.zeros([self.n_x, self.channel, self.h_out, self.w_out])

        
        # instance of Img2colIndices
        #self.img2col_indices = Img2colIndices((self.h_filter, self.w_filter), self.padding, self.stride)
        #time_step的设置也很有学问，对于泄露值以及阈值的作用也不一样，多timestep仿真，馈送的是zeros

        return self.forward(X1, X2, time_step)
    
    def forward(self, X1, X2, time_step):

        #脉冲相加，还是膜电位相加
        out = X1 + X2
        
        # membrane potential accumulation
        self.mem = self.mem + out

        if self.bn == 1:
            # leak
            self.mem = self.mem + np.repeat(np.repeat(np.repeat(self.leakages[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0)
            # generate spikes (scatter-and-gather)   
            if self.noise_ratio > 0:
               for i in range((2**self.precisions[0])*self.precisions[1]):
                   mask = np.where(np.random.rand(self.mem.shape[0], self.mem.shape[1], self.mem.shape[2], self.mem.shape[3]) > self.noise_ratio, 1, 0) 
                   self.spike_out = self.spike_out + np.where(self.mem >= np.repeat(np.repeat(np.repeat(self.thresholds[i,:][:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                   self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), 1, 0) * mask
            else:
                for i in range((2**self.precisions[0])*self.precisions[1]):
                   self.spike_out = self.spike_out + np.where(self.mem >= np.repeat(np.repeat(np.repeat(self.thresholds[i,:][:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                   self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), 1, 0)

            # count spikes
            self.spike_num = self.spike_num + np.sum(self.spike_out)
            # count synaptic operations for snn
            #可以忽略不计
            self.sop_num = self.sop_num + 1
            #self.mem_trace[t] = self.mem_trace + np.abs(self.x_col)
            self.neurons = self.spike_out.shape[0] * self.spike_out.shape[1] * self.spike_out.shape[2] * self.spike_out.shape[3] * (2**self.precisions[0]) * self.precisions[1]
        
            #只产生膜电位，而不输出脉冲的情况，不应该打印
            print('Time_step: ', time_step, 'Layer: ', self.layer_num, ', spiking rate: ', np.sum(self.spike_out)/self.neurons)
        else:
            #这里的无bn跟神经网络里面的没有bn训练不是一回事，而是代表这里直接输出膜电位，对于conv2d_add层来说，一般是需要bn的
            #无论有没有bn，主要的sop都在卷积这里，不在bn也不在其它层，如conv2d_add，尤其是在脉冲计算领域（与多位宽实数值计算的优缺点，有待比较）
            # count spikes
            self.spike_num = self.spike_num + np.sum(self.spike_out)
            # count synaptic operations for snn
            #可以忽略不计
            self.sop_num = self.sop_num + 1
            #self.mem_trace[t] = self.mem_trace + np.abs(self.x_col)
            self.neurons = self.spike_out.shape[0] * self.spike_out.shape[1] * self.spike_out.shape[2] * self.spike_out.shape[3] * (2**self.precisions[0]) * self.precisions[1]
            self.spike_out = self.mem
        
        #对bn=0，这个脉冲数以及sop数没有意义
        return self.spike_out, self.spike_num, self.sop_num





class BatchNorm2d():
    """
    batch normalization instance
    """
    count = 0
    def __init__(self, n_channel, momentum):
        """
        parameters:
            n_channel: number of input feature channle
            momentum: moving_mean/moving_var momentum term
        """
        BatchNorm2d.count += 1
        self.layer_num = BatchNorm2d.count

        self.n_channel = n_channel
        self.momentum = momentum
        
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = np.ones((1, n_channel, 1, 1))
        self.beta = np.zeros((1, n_channel, 1, 1))
        
        # initialization
        self.moving_mean = np.zeros((1, n_channel, 1, 1))
        self.moving_var = np.zeros((1, n_channel, 1, 1))
        
        self.params = [self.gamma, self.beta]
        self.debug = 10
        self.type = 'bn'
    
    def __call__(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: train or test
        """
        self.X = X  
        return self.forward(X, time_step, mode)
    
    def forward(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: mode: train or test
        """
        if mode != 'train':
            self.x_norm = (X - self.moving_mean) / np.sqrt(self.moving_var + 1e-5)
        else:
            # computing with multicast
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            self.var = X.var(axis=(0, 2, 3), keepdims=True)  
            self.x_norm = (X - mean) / (np.sqrt(self.var + 1e-5))
            
            # update moving_mean/moving_var
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        # gamma and beta term
        out = self.x_norm * self.gamma + self.beta
        return out


class BatchNorm2d_shortcut():
    """
    batch normalization instance
    """
    count = 0
    def __init__(self, n_channel, momentum):
        """
        parameters:
            n_channel: number of input feature channle
            momentum: moving_mean/moving_var momentum term
        """
        BatchNorm2d.count += 1
        self.layer_num = BatchNorm2d.count

        self.n_channel = n_channel
        self.momentum = momentum
        
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = np.ones((1, n_channel, 1, 1))
        self.beta = np.zeros((1, n_channel, 1, 1))
        
        # initialization
        self.moving_mean = np.zeros((1, n_channel, 1, 1))
        self.moving_var = np.zeros((1, n_channel, 1, 1))
        
        self.params = [self.gamma, self.beta]
        self.debug = 10
        self.type = 'bn_shortcut'
    
    def __call__(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: train or test
        """
        self.X = X  
        return self.forward(X, time_step, mode)
    
    def forward(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: mode: train or test
        """
        if mode != 'train':
            self.x_norm = (X - self.moving_mean) / np.sqrt(self.moving_var + 1e-5)
        else:
            # computing with multicast
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            self.var = X.var(axis=(0, 2, 3), keepdims=True)  
            self.x_norm = (X - mean) / (np.sqrt(self.var + 1e-5))
            
            # update moving_mean/moving_var
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        # gamma and beta term
        out = self.x_norm * self.gamma + self.beta
        return out        
    

class Flatten():
    """
    flatten function
    """
    count = 0
    def __init__(self):
        Flatten.count += 1
        self.layer_num = Flatten.count
        self.debug = 10
        self.type = 'flatten'

    def __call__(self, X, time_step):
        self.x_shape = X.shape # (batch_size, channels, height, width)
        
        return self.forward(X, time_step)
    
    def forward(self, X, time_step):
        out = X.transpose(0, 2, 3, 1)
        out = out.ravel().reshape(self.x_shape[0], -1)
        return out
    



class Linear():
    """
    linear layer or fully connected layer
    """
    count = 0
    def __init__(self, dim_in, dim_out, use_ternary):
        """
        parameters：
            dim_in: input size
            dim_out: output size
        """
        Linear.count += 1
        self.layer_num = Linear.count

        # initialization
        scale = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
        self.bias = np.zeros(dim_out)
        # self.weight = np.random.randn(dim_in, dim_out)
        # self.bias = np.zeros(dim_out)
        
        self.params = [self.weight, self.bias]
        self.debug = 10
        self.type = 'fc'
        self.use_ternary = use_ternary
        
    def __call__(self, X, time_step):
        """
        parameters：
            X：input, shape=(batch_size, dim_in)
        return：
            xw + b
        """
        self.X = X
        return self.forward(time_step)
    
    def forward(self, time_step):
        return np.dot(self.X, self.weight) + self.bias
    

class Relu(object):
    """
    relu function
    """
    def __init__(self):
        self.X = None
    
    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return np.maximum(0, X)
    

