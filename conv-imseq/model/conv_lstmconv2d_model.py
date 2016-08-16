import os
import sys
# add path to libraries for ipython
sys.path.append(os.path.expanduser("~/libs"))
import tensortools as tt

import tensorflow as tf


def _encoder(frame_input, reg_lambda):  
    # conv1  
    conv1 = tt.network.conv2d("conv1", frame_input,
                              128, (9, 9), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv1)
    
    # conv2  
    conv2 = tt.network.conv2d("conv2", conv1,
                              128, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv2)
    
    # conv3  
    conv3 = tt.network.conv2d("conv3", conv2,
                              128, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.tanh) # a paper proposes to use TANH here
    tt.board.activation_summary(conv3)
        
    return conv3


def _decoder(rep_input, reg_lambda, channels):
    conv1t = tt.network.conv2d_transpose("deconv1", rep_input,
                                         128, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv1t)
    
    conv2t = tt.network.conv2d_transpose("deconv2", conv1t,
                                         128, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv2t)
    
    conv3t = tt.network.conv2d_transpose("deconv3", conv2t,
                                         channels, (9, 9), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))
    tt.board.activation_summary(conv3t)
        
    return conv3t


def _conv2d_lstm(conv_image_sequence,
                 layers, filters, height, width, ksize_input, ksize_hidden):   
    lstm_cell = tt.recurrent.BasicLSTMConv2DCell(height, width, filters,
                                                 ksize_input=ksize_input,
                                                 ksize_hidden=ksize_hidden,
                                                 device='/cpu:0')
    if layers > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * layers)

    # Get lstm cell output
    outputs, states = tt.recurrent.rnn_conv2d(lstm_cell, conv_image_sequence)
    return outputs[-1]    


class ConvLSTMConv2DModel(tt.model.AbstractModel):
    """Simple model that uses a convolutional autoencoder architecture.
       It has to learn the temporal dependencies by its own.
       This model only predicts a single frame.
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4,
                 lstm_layers=1, lstm_ksize_input=(5, 5), lstm_ksize_hidden=(7,7)):
        self._lstm_layers = lstm_layers
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        super(ConvLSTMConv2DModel, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        input_seq = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, self.input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]

        # encoder
        with tf.variable_scope("encoder"):
            convolved_inputs = []
            for i, input_part in enumerate(input_seq):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                convolved_input = _encoder(input_part, self.reg_lambda)
                convolved_inputs.append(convolved_input)

        # lstm
        with tf.variable_scope('lstm'):
            lstm_input_shape = convolved_inputs[0].get_shape().as_list()
            lstm_output = _conv2d_lstm(convolved_inputs,
                                       self._lstm_layers, lstm_input_shape[3],
                                       lstm_input_shape[1], lstm_input_shape[2],
                                       self._lstm_ksize_input, self._lstm_ksize_hidden)

        # decoder
        with tf.variable_scope('decoder'):
            prediction = _decoder(lstm_output, self.reg_lambda, self.input_shape[4])
        return prediction

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
