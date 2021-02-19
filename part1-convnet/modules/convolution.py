import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        stride = self.stride
        pad = self.padding
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        H_n = int(1 + (H + 2 * pad - HH) / stride)
        W_n = int(1 + (W + 2 * pad - WW) / stride)
        X_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        out = np.zeros((N, F, H_n, W_n))
        for i in range(N):
            for j in range(H_n):
                for k in range(W_n):
                    for f in range(F):
                        X_i = X_pad[i]
                        inp_con = X_i[:, j*stride:j*stride+HH, k*stride:k*stride+WW]
                        out_con = (inp_con*self.weight[f,:,:,:]).sum() + self.bias[f]
                        out[i, f, j, k] = out_con
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape
        stride = self.stride
        pad = self.padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        H_n = 1 + (H + 2 * pad - HH) // stride
        W_n = 1 + (W + 2 * pad - WW) // stride
        dx_pad = np.zeros_like(x_pad)
        dx = np.zeros_like(x)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)
        for n in range(N):
            for f in range(F):
                db[f] += dout[n, f].sum()
                for j in range(0, H_n):
                    for i in range(0, W_n):
                        dw[f] += x_pad[n,:,j*stride:j*stride+HH,i*stride:i*stride+WW]*dout[n,f,j,i]
                        dx_pad[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW] += self.weight[f]*dout[n, f, j, i]
        dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################