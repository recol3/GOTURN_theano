from __future__ import division

import os
import numpy as np
import theano
import theano.tensor as T

import Layers
import lasagne.updates
from lasagne.layers import normalization


def preprocess(input, mean=(104, 117, 123)):
    # Subtracts mean image value as given in regressor.cpp(85) (BGR order)
    mean_sub = np.empty((1, 3, 1, 1), dtype=theano.config.floatX)
    mean_sub[0, :, 0, 0] = mean
    return input - mean_sub


class ConvLayers(object):
    def __init__(self, input, input_shape, rng=None, Ws=None, bs=None):

        assert(
            (rng is     None and Ws is     None and bs is     None) or
            (rng is not None and Ws is     None and bs is     None) or
            (rng is     None and Ws is not None and bs is not None)
        )

        if Ws is None and bs is None:
            Ws = [None] * 5
            bs = [None] * 5

        self.trainable_layers = []


        ############################## conv1 ##############################

        self.layer_conv1 = Layers.ConvLayer(
            input=input, 
            input_shape=input_shape, 
            filter_shape=(96, 3, 11, 11), 
            stride=(4, 4), 
            border_mode='valid', 
            activation=T.nnet.relu, 
            rng=rng, 
            W=Ws[0], 
            b=bs[0]
        )
        conv1_output_shape = (None, 96, 55, 55)  # (227 - 11) / 4 + 1 = 55
        self.trainable_layers.append(self.layer_conv1)

        self.layer_pool1 = Layers.PoolLayer(
            input=self.layer_conv1.output, 
            kernel_size=(3, 3), 
            stride=(2, 2)
        )
        pool1_output_shape = (None, 96, 27, 27)  # (55 - 3) / 2 + 1 = 27

        self.layer_norm1 = normalization.LocalResponseNormalization2DLayer(
            incoming=pool1_output_shape,  # Actual input is param to .get_output_for()
            # alpha=0.0001, 
            alpha=0.0001/5,  # Caffe documentation uses alpha/n as coeff of summation
            k=1,  # Not specified; Caffe says default is 1, but AlexNet paper says 2
            beta=0.75, 
            n=5
        )
        self.layer_norm1.output = self.layer_norm1.get_output_for(self.layer_pool1.output)
        self.layer_norm1.output_group1 = self.layer_norm1.output[:, :48, :, :]
        self.layer_norm1.output_group2 = self.layer_norm1.output[:, 48:, :, :]
        norm1_group1_output_shape = (None, 48, 27, 27)
        norm1_group2_output_shape = (None, 48, 27, 27)


        ############################## conv2 ##############################
        self.layer_conv2_group1 = Layers.ConvLayer(
            input=self.layer_norm1.output_group1, 
            input_shape=norm1_group1_output_shape, 
            filter_shape=(128, 48, 5, 5), 
            stride=(1, 1), 
            border_mode=(2, 2),  # pad by 2 pixels to each edge, so height and width += 4
            activation=T.nnet.relu, 
            rng=rng, 
            W=Ws[1][:128] if Ws[1] is not None else None,
            b=bs[1][:128] if bs[1] is not None else None
        )
        conv2_group1_output_shape = (None, 128, 27, 27)  # (27 + 4 - 5) / 1 + 1 = 27
        self.trainable_layers.append(self.layer_conv2_group1)

        self.layer_conv2_group2 = Layers.ConvLayer(
            input=self.layer_norm1.output_group2, 
            input_shape=norm1_group2_output_shape, 
            filter_shape=(128, 48, 5, 5), 
            stride=(1, 1), 
            border_mode=(2, 2),  # pad by 2 pixels to each edge, so height and width += 4
            activation=T.nnet.relu, 
            rng=rng, 
            W=Ws[1][128:] if Ws[1] is not None else None,
            b=bs[1][128:] if bs[1] is not None else None
        )
        conv2_group2_output_shape = (None, 128, 27, 27)  # (27 + 4 - 5) / 1 + 1 = 27
        self.trainable_layers.append(self.layer_conv2_group2)

        self.layer_pool2_group1 = Layers.PoolLayer(
            input=self.layer_conv2_group1.output, 
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        pool2_group1_output_shape = (None, 128, 13, 13)  # (27 - 3) / 2 + 1 = 13

        self.layer_pool2_group2 = Layers.PoolLayer(
            input=self.layer_conv2_group2.output, 
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        pool2_group2_output_shape = (None, 128, 13, 13)  # (27 - 3) / 2 + 1 = 13

        self.layer_norm2_group1 = normalization.LocalResponseNormalization2DLayer(
            incoming=pool2_group1_output_shape,  # Actual input is param to .get_output_for()
            # alpha=0.0001, 
            alpha=0.0001/5,  # Caffe documentation uses alpha/n as coeff of summation
            k=1,  # Not specified; Caffe says default is 1
            beta=0.75, 
            n=5
        )
        self.layer_norm2_group1.output = self.layer_norm2_group1.get_output_for(self.layer_pool2_group1.output)
        norm2_group1_output_shape = pool2_group1_output_shape

        self.layer_norm2_group2 = normalization.LocalResponseNormalization2DLayer(
            incoming=pool2_group2_output_shape,  # Actual input is param to .get_output_for()
            # alpha=0.0001, 
            alpha=0.0001/5,  # Caffe documentation uses alpha/n as coeff of summation
            k=1,  # Not specified; Caffe says default is 1
            beta=0.75, 
            n=5
        )
        self.layer_norm2_group2.output = self.layer_norm2_group2.get_output_for(self.layer_pool2_group2.output)
        norm2_group2_output_shape = pool2_group2_output_shape


        ############################## conv3 ##############################

        norm2_grouped_output = T.concatenate((self.layer_norm2_group1.output, self.layer_norm2_group2.output), axis=1)
        norm2_grouped_output_shape = (None, 256, 13, 13)

        self.layer_conv3 = Layers.ConvLayer(
            input=norm2_grouped_output, 
            input_shape=norm2_grouped_output_shape, 
            filter_shape=(384, 256, 3, 3), 
            stride=(1, 1),
            border_mode=(1, 1),
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[2],
            b=bs[2]
        )
        conv3_output_shape = (None, 384, 13, 13)  # (13 + 2 - 3) / 1 + 1 = 13
        self.trainable_layers.append(self.layer_conv3)
        self.layer_conv3.output_group1 = self.layer_conv3.output[:, :192, :, :]
        self.layer_conv3.output_group2 = self.layer_conv3.output[:, 192:, :, :]
        conv3_group1_output_shape = (None, 192, 13, 13)
        conv3_group2_output_shape = (None, 192, 13, 13)


        ############################## conv4 ##############################

        self.layer_conv4_group1 = Layers.ConvLayer(
            input=self.layer_conv3.output_group1, 
            input_shape=conv3_group1_output_shape, 
            filter_shape=(192, 192, 3, 3), 
            stride=(1, 1),
            border_mode=(1, 1),
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[3][:192] if Ws[3] is not None else None,
            b=bs[3][:192] if bs[3] is not None else None
        )
        conv4_group1_output_shape = (None, 192, 13, 13)  # (13 + 2 - 3) / 1 + 1 = 13
        self.trainable_layers.append(self.layer_conv4_group1)

        self.layer_conv4_group2 = Layers.ConvLayer(
            input=self.layer_conv3.output_group2, 
            input_shape=conv3_group2_output_shape, 
            filter_shape=(192, 192, 3, 3), 
            stride=(1, 1),
            border_mode=(1, 1),
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[3][192:] if Ws[3] is not None else None,
            b=bs[3][192:] if bs[3] is not None else None
        )
        conv4_group2_output_shape = (None, 192, 13, 13)  # (13 + 2 - 3) / 1 + 1 = 13
        self.trainable_layers.append(self.layer_conv4_group2)


        ############################## conv5 ##############################

        self.layer_conv5_group1 = Layers.ConvLayer(
            input=self.layer_conv4_group1.output, 
            input_shape=conv4_group1_output_shape,
            filter_shape=(128, 192, 3, 3), 
            stride=(1, 1), 
            border_mode=(1, 1), 
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[4][:128] if Ws[4] is not None else None,
            b=bs[4][:128] if bs[4] is not None else None
        )
        conv5_group1_output_shape = (None, 128, 13, 13)  # (13 + 2 - 3) / 1 + 1 = 13
        self.trainable_layers.append(self.layer_conv5_group1)

        self.layer_pool5_group1 = Layers.PoolLayer(
            input=self.layer_conv5_group1.output, 
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        pool5_group1_output_shape = (None, 128, 6, 6)  # (13 - 3) / 2 + 1 = 6

        self.layer_conv5_group2 = Layers.ConvLayer(
            input=self.layer_conv4_group2.output, 
            input_shape=conv4_group2_output_shape,
            filter_shape=(128, 192, 3, 3), 
            stride=(1, 1), 
            border_mode=(1, 1), 
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[4][128:] if Ws[4] is not None else None,
            b=bs[4][128:] if bs[4] is not None else None
        )
        conv5_group2_output_shape = (None, 128, 13, 13)  # (13 + 2 - 3) / 1 + 1 = 13
        self.trainable_layers.append(self.layer_conv5_group2)

        self.layer_pool5_group2 = Layers.PoolLayer(
            input=self.layer_conv5_group2.output, 
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        pool5_group2_output_shape = (None, 128, 6, 6)  # (13 - 3) / 2 + 1 = 6

        pool5_grouped_output = T.concatenate((self.layer_pool5_group1.output, self.layer_pool5_group2.output), axis=1)

        self.output = pool5_grouped_output
        self.trainable_params = [param for layer in self.trainable_layers for param in layer.params]


class FCLayers(object):
    def __init__(self, training_enabled, target_convLayers_output, curFrame_convLayers_output, rng=None, Ws=None,
                 bs=None, dropout_scale_like_caffe=False, dropout_p=0.5):

        assert(
            (rng is     None and Ws is     None and bs is     None) or
            (rng is not None and Ws is     None and bs is     None) or
            (rng is     None and Ws is not None and bs is not None)
        )

        # Default fill values from GOTURN's tracker.prototxt (stdev for Ws, constant for bs)
        if Ws is None and bs is None:
            Ws = [0.005, 0.005, 0.005, 0.01]
            bs = [ np.ones(4096), np.ones(4096), np.ones(4096), np.zeros(4) ]

        self.trainable_layers = []

        self.flat_input = T.concatenate( 
            (
                # Flatten the (batch_size, 256, 6, 6)-shape tensors into batch_size rows, then transpose them to
                # columns, then stack vertically
                # theano.tensor.reshape behaves like numpy.reshape, reading elements with the last index changing
                # fastest (C-like)
                T.reshape(target_convLayers_output, (-1, 256*6*6)).transpose(),
                T.reshape(curFrame_convLayers_output, (-1, 256*6*6)).transpose()
            ), 
            axis=0)


        self.layer_fc6new = Layers.FCLayer(
            input=self.flat_input, 
            n_in=256*6*6 * 2, 
            n_out=4096, 
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[0],
            b=bs[0]
        )
        self.trainable_layers.append(self.layer_fc6new)

        self.layer_drop6 = Layers.DropoutLayer(
            input=self.layer_fc6new.output, 
            training_enabled=training_enabled,
            p=dropout_p,
            rng=rng,
            scale_like_caffe=dropout_scale_like_caffe
        )


        self.layer_fc7new = Layers.FCLayer(
            input=self.layer_drop6.output, 
            n_in=4096, 
            n_out=4096, 
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[1],
            b=bs[1]
        )
        self.trainable_layers.append(self.layer_fc7new)

        self.layer_drop7 = Layers.DropoutLayer(
            input=self.layer_fc7new.output, 
            training_enabled=training_enabled,
            p=dropout_p,
            rng=rng,
            scale_like_caffe=dropout_scale_like_caffe
        )


        self.layer_fc7newb = Layers.FCLayer(
            input=self.layer_drop7.output, 
            n_in=4096, 
            n_out=4096, 
            activation=T.nnet.relu,
            rng=rng,
            W=Ws[2],
            b=bs[2])
        self.trainable_layers.append(self.layer_fc7newb)

        self.layer_drop7b = Layers.DropoutLayer(
            input=self.layer_fc7newb.output,
            training_enabled=training_enabled, 
            p=dropout_p,
            rng=rng,
            scale_like_caffe=dropout_scale_like_caffe
        )


        self.layer_fc8shapes = Layers.FCLayer(
            input=self.layer_drop7b.output, 
            n_in=4096, 
            n_out=4, 
            activation=None, # TODO activation?
            rng=rng,
            W=Ws[3],
            b=bs[3]
        )
        self.trainable_layers.append(self.layer_fc8shapes)

        self.output = self.layer_fc8shapes.output
        self.trainable_params = [param for layer in self.trainable_layers for param in layer.params]


class Network(object):
    def __init__(self, 
        target_input_shape=(None, 3, 227, 227),
        curFrame_input_shape=(None, 3, 227, 227),
        curFrame_bbox_shape=(4, None),
        rng=None,
        target_conv_Ws=None,
        target_conv_bs=None,
        curFrame_conv_Ws=None,
        curFrame_conv_bs=None,
        fc_Ws=None,
        fc_bs=None,
        dropout_scale_like_caffe=False,
        dropout_p=0.5
    ):

        # target_input_shape and curFrame_input_shape: (batch_size, num_channels, height, width).
        # Passed to T.nnet.conv2d's input_shape parameter. Any element of that parameter can be None to indicate to
        # conv2d that that element is not known at compile time.
        self.target_input_shape = target_input_shape
        self.curFrame_input_shape = curFrame_input_shape
        self.curFrame_bbox_shape = curFrame_bbox_shape

        # Symbolic variabels that can be accessed later. 4-way tensors: (images, channels, height, width)
        self.target_input = T.tensor4(name="target_input")
        self.curFrame_input = T.tensor4(name="curFrame_input")
        self.curFrame_bbox = T.matrix(name="curFrame_bbox")
        self.training_enabled = T.iscalar(name="training_enabled")
        self.target_input_prep = preprocess(self.target_input)
        self.curFrame_input_prep = preprocess(self.curFrame_input)

        self.trained_iters = theano.shared(value=0, name="trained_iters", borrow=True)

        assert(
            (
                rng              is     None and 
                target_conv_Ws   is     None and 
                target_conv_bs   is     None and 
                curFrame_conv_Ws is     None and 
                curFrame_conv_bs is     None and 
                fc_Ws            is     None and 
                fc_bs            is     None
            ) or
            (
                rng              is not None and 
                target_conv_Ws   is     None and 
                target_conv_bs   is     None and 
                curFrame_conv_Ws is     None and 
                curFrame_conv_bs is     None and 
                fc_Ws            is     None and 
                fc_bs            is     None
            ) or
            (
                rng              is     None and 
                target_conv_Ws   is not None and 
                target_conv_bs   is not None and 
                curFrame_conv_Ws is not None and 
                curFrame_conv_bs is not None and 
                fc_Ws            is not None and 
                fc_bs            is not None
            )
        )

        self.target_convLayers = ConvLayers(
            input=self.target_input_prep,
            input_shape=self.target_input_shape,
            rng=rng,
            Ws=target_conv_Ws,
            bs=target_conv_bs
        )
        self.curFrame_convLayers = ConvLayers(
            input=self.curFrame_input_prep,
            input_shape=self.curFrame_input_shape,
            rng=rng,
            Ws=curFrame_conv_Ws,
            bs=curFrame_conv_bs
        )
        self.fcLayers = FCLayers(
            training_enabled=self.training_enabled,
            target_convLayers_output=self.target_convLayers.output, 
            curFrame_convLayers_output=self.curFrame_convLayers.output,
            rng=rng,
            Ws=fc_Ws,
            bs=fc_bs,
            dropout_scale_like_caffe=dropout_scale_like_caffe,
            dropout_p=dropout_p
        )

        self.output = self.fcLayers.output

        self.target_conv_params = self.target_convLayers.trainable_params
        self.curFrame_conv_params = self.curFrame_convLayers.trainable_params
        self.fcLayers_params = self.fcLayers.trainable_params

        self.trainable_layers = self.fcLayers.trainable_layers  # Conv layer params are fixed
        self.trainable_params = self.fcLayers_params

        self.bbox_diff = self.output - self.curFrame_bbox
        self.bbox_l1_errs = self.bbox_diff.norm(1, axis=0)
        self.bbox_l1_err = self.bbox_diff.norm(1, axis=0).sum()


    def build_train_func(self, optim="sgdm", opts=dict(), profile_on=False):
        if optim == "sgdm":
            return self.build_train_func_sgdm(opts, profile_on)
        elif optim == "adam":
            return self.build_train_func_adam(opts, profile_on)
        else:
            raise NotImplementedError


    def build_train_func_adam(self, opts=dict(), profile_on=False):
        # Default learning rate for Lasagne is 0.001 (https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py)
        learning_rate = opts.get("learning_rate", 0.000001)
        # Default beta1 from Adam ref (https://arxiv.org/pdf/1412.6980v8.pdf) and Lasagne
        beta1 = opts.get("beta1", 0.9)
        # Default beta2 from Adam ref and Lasagne
        beta2 = opts.get("beta2", 0.999)
        # Default epsilon from Adam ref and Lasagne
        epsilon = opts.get("epsilon", 1e-8)
        # Weight decay rate from GOTURN sgdm (see build_train_func_sgdm)
        weight_decay_rate = opts.get("weight_decay_rate", 0.0005)
        # Decay multipliers from GOTURN sgdm (see build_train_func_sgdm)
        fc_decay_mults = opts.get("fc_decay_mults", [1, 0]*4)
        l2_penalty = self.compute_l2_penalty(weight_decay_rate, fc_decay_mults)
        cost = self.bbox_l1_err + l2_penalty
        updates = lasagne.updates.adam(loss_or_grads=cost, params=self.trainable_params, learning_rate=learning_rate,
                                       beta1=beta1, beta2=beta2, epsilon=epsilon)
        updates[self.trained_iters] = self.trained_iters + 1
        train_func = self.compile_train_func(updates, cost=cost, penalty=l2_penalty, profile_on=profile_on)
        return train_func


    def build_train_func_sgdm(self, opts=dict(), profile_on=False):

        # Defaults based on the paper and GOTURN's solver.prototxt
        # Learning rate: GOTURN paper (p. 757, top) says 1e-5, but solver.prototxt has base_lr: 0.000001. It's probably
        # because he sets lr_mult to 10 for the FC layer weights (20 for biases) in tracker.prototxt.
        learning_rate           = opts.get("learning_rate", 0.000001)
        fc_lr_mults             = opts.get("fc_lr_mults", [10, 20]*4)
        learning_decay_stepsize = opts.get("learning_decay_stepsize", 100000)
        gamma                   = opts.get("gamma", 0.1)
        momentum                = opts.get("momentum", 0.9)
        weight_decay_rate       = opts.get("weight_decay_rate", 0.0005)
        fc_decay_mults          = opts.get("fc_decay_mults", [1, 0]*4)
        # fc_lr_mults and fc_decay_mults: lists with two values per trainable layer (for W followed by for b)

        assert(len(fc_lr_mults) == len(self.trainable_params))

        # Learning rate decay (see Caffe's caffe.proto lines 157-71)
        learning_rate_dec = learning_rate * gamma ** T.floor(self.trained_iters / learning_decay_stepsize)
        learning_rates = [learning_rate_dec * lr_mult for lr_mult in fc_lr_mults] 

        l2_penalty = self.compute_l2_penalty(weight_decay_rate, fc_decay_mults)

        cost = self.bbox_l1_err + l2_penalty

        param_gradients = [T.grad(cost, param) for param in self.trainable_params]
        
        updates = dict([  # dict() automatically converts list of pairs into dictionary
            (param, param - lr * g)
            for param, lr, g in zip(self.trainable_params, learning_rates, param_gradients)
        ])

        # Momentum (see Caffe's caffe.proto line 175)
        updates = lasagne.updates.apply_momentum(updates=updates, momentum=momentum)

        # Add +1 to iteration count to updates (for learning rate decay)
        updates[self.trained_iters] = self.trained_iters + 1

        # Compile function
        # TODO check how much faster using givens and a symbolic index is and consider doing that instead
        train_func = self.compile_train_func(updates, cost=cost, penalty=l2_penalty, profile_on=profile_on)

        return train_func


    def compute_l2_penalty(self, weight_decay_rate, fc_decay_mults):
        # Weight decay (see Caffe's caffe.proto lines 176-9)
        assert(len(fc_decay_mults) == len(self.trainable_params))
        weight_decay_rates = [weight_decay_rate * decay_mult for decay_mult in fc_decay_mults]
        l2_penalty = sum([dr * param.norm(L=2)**2 for param, dr in zip(self.trainable_params, weight_decay_rates)])
        # Note: the above takes the squared L2 norm for each W and b, multiplies by the corresponding decay coefficient,
        # them sums them all. Unclear what Caffe does
        return l2_penalty


    # TODO? make these not instance functions
    def compile_train_func(self, updates, cost=None, penalty=None, profile_on=False):
        # cost and penalty are just for function output; not necessary

        if profile_on:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # TODO check how much faster using givens and a symbolic index is and consider doing that instead
        train_func = theano.function(
            inputs=[self.target_input, self.curFrame_input, self.curFrame_bbox],
            outputs=[self.output, cost, self.bbox_l1_err, penalty],
            updates=updates,
            givens={
                # Always true during training, so we hardcode to 1 and substitute it in here
                self.training_enabled : np.cast['int32'](1) 
            },
            profile=profile_on
        )
        return train_func


    def build_test_func(self):
        test_model = theano.function(
            inputs=[self.target_input, self.curFrame_input],
            outputs=self.output,
            givens={
                # Always false during testing, so we hardcode to 0 and substitute it in here
                self.training_enabled : np.cast['int32'](0) 
            }
        )
        return test_model


    def build_val_func(self, weight_decay_rate=0.0005, fc_decay_mults=[1, 0]*4):

        l2_penalty = self.compute_l2_penalty(weight_decay_rate, fc_decay_mults)

        cost = self.bbox_l1_err + l2_penalty

        # Validation function: same as build_test_func except takes true bbox and returns error; same as build_train_
        # func except no update
        val_func = theano.function(
            inputs=[self.target_input, self.curFrame_input, self.curFrame_bbox],
            outputs=[self.output, cost, self.bbox_l1_err, l2_penalty],
            givens={
                # Always false during validation, so we hardcode to 0 and substitute it in here
                self.training_enabled : np.cast['int32'](0)
            }
        )
        return val_func


    def init_tensors(self, batch_size):
        targets_shape   = (batch_size,) + self.target_input_shape[1:]
        curFrames_shape = (batch_size,) + self.curFrame_input_shape[1:]
        bboxes_shape    = self.curFrame_bbox_shape[:-1] + (batch_size,)
        # output's shape is (1, 1, 4, batch_size), so match this.

        targets         = np.empty(shape=targets_shape,   dtype=theano.config.floatX)
        curFrames       = np.empty(shape=curFrames_shape, dtype=theano.config.floatX)
        curFrame_bboxes = np.empty(shape=bboxes_shape,    dtype=theano.config.floatX)

        return targets, curFrames, curFrame_bboxes
