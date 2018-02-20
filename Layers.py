from __future__ import division

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# Much of this file is taken from http://deeplearning.net/tutorial/lenet.html


class ConvLayer(object):
    """Convolutional layer without pooling """

    def __init__(self, input, input_shape, filter_shape, stride, border_mode, activation=T.nnet.relu, rng=None, W=None,
                 b=None):
        """
        Allocate a ConvLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape input_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)

        :type input_shape: tuple or list of length 4
        :param input_shape: (batch size, num input feature maps (or channels), image height, image width)
        """

        assert(
            (rng is     None and W is     None and b is     None) or
            (rng is not None and W is     None and b is     None) or
            (rng is     None and W is not None and b is not None)
        )
        assert input_shape[1] == filter_shape[1]
        self.input = input

        if W is not None and b is not None:
            assert(W.shape == filter_shape)
            assert(b.shape[0] == filter_shape[0])
            W_values = W.astype(theano.config.floatX)
            b_values = b.astype(theano.config.floatX)
        else:
            rng = rng if rng else np.random.RandomState()
 
            # initialize weights randomly
            W_std = 0.01
            W_values = np.asarray(
                rng.normal(loc=0.0, scale=W_std, size=filter_shape),
                dtype=theano.config.floatX
            )
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            input_shape=input_shape,
            filter_shape=filter_shape,
            border_mode=border_mode,
            subsample=stride,
            filter_flip=False  # Caffe
        )

        # Add the bias term. Since the bias is a vector (1D array), we first reshape it to a tensor of shape (1,
        # n_filters, 1, 1). Each bias will thus be broadcasted across mini-batches and feature map width & height
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class PoolLayer(object):
    """Pooling-only layer """

    def __init__(self, input, kernel_size=(3, 3), stride=(2, 2)):

        # 3x3 kernel with stride of 2 -> (dimension - 3) / 2 + 1

        pooled_out = pool.pool_2d(
            input=input,
            ds=kernel_size,
            ignore_border=True,
            st=stride
        )

        self.output = pooled_out

        self.input = input


class FCLayer(object):

    def __init__(self, input, n_in, n_out, activation=T.nnet.relu, rng=None, W=0.01, b=None):
        # Input W must be either a matrix of predefined weights or a scalar indicating the standard deviation of the
        # normal distribution to draw from when filling weights randomly
        # Input b must be either a vector of predefined biases or None; if None, biases are initialized to 0s

        self.input = input

        if rng is None:
            rng = np.random.RandomState()
        else:
            # If rng is supplied, at least one of W or b must not be supplied
            assert(np.isscalar(W) or b is None)

        if not np.isscalar(W):
            assert(len(W.shape) == 2)
            assert(W.shape[0] == n_out)
            assert(W.shape[1] == n_in)
            W_values = W.astype(theano.config.floatX)
        else:
            # initialize weights randomly
            W_std = W
            W_values = np.asarray(
                rng.normal(loc=0.0, scale=W_std, size=(n_out, n_in)),
                dtype=theano.config.floatX
            )

        if b is not None:
            assert(b.shape[0] == n_out or (b.shape[0] == 1 and b.shape[1] == n_out))
            b_values = b.astype(theano.config.floatX)
        else:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)


        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values.reshape((-1, 1)), name='b', borrow=True, broadcastable=(False, True))
        # reshape is necessary to make b an n x 1 "matrix" (column) rather than a 1D vector so that it can be added to
        # the output, which is columns. broadcastable is necessary to explicitly tell Theano b can be broadcast across
        # columns.
        # TODO? Consider switching to row vectors and pre-multiplying input to avoid this

        lin_output = T.dot(self.W, input) + self.b  # output as column vectors
        # TODO? if switch order, don't forget to change W shape asserts above
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]


class DropoutLayer(object):

    def __init__(self, input, training_enabled, p=0.5, rng=None, scale_like_caffe=False):
        # p is probability of NOT dropping a unit

        if rng is None:
            rng = np.random.RandomState()
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)

        # Caffe scales *up* the dropout output by 1/(1-p) during training, and passes it through unchanged during
        # testing: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp(17, 41, 44). But that
        # seems to give me poorer performance.
        self.scale_like_caffe = scale_like_caffe  # Save this
        self.dropout_p = p
        if scale_like_caffe:
            train_output = mask * input * 1/(1-p) if p is not 1 else input
            test_output = input
        else:
            train_output = mask * input
            test_output = p * input

        self.output = T.switch(T.neq(training_enabled, 0), train_output, test_output)
