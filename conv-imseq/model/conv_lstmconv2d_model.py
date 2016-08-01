import os
import sys
# add path to libraries for ipython
sys.path.append(os.path.expanduser("~/libs"))
import tensortools as tt

import tensorflow as tf


def encoder(frame_input, LAMBDA):  
    # conv1  
    conv1 = tt.network.conv2d("conv1", frame_input,
                              128, 9, 9, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv1)
    
    # conv2  
    conv2 = tt.network.conv2d("conv2", conv1,
                              128, 5, 5, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv2)
    
    # conv3  
    conv3 = tt.network.conv2d("conv3", conv2,
                              128, 5, 5, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.tanh) # a paper proposes to use TANH here
    tt.board.activation_summary(conv3)
        
    return conv3


def decoder(rep_input, FRAME_CHANNELS, LAMBDA):
    conv1t = tt.network.conv2d_transpose("deconv1", rep_input,
                                         128, 5, 5, 2, 2,
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv1t)
    
    conv2t = tt.network.conv2d_transpose("deconv2", conv1t,
                                         128, 5, 5, 2, 2,
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv2t)
    
    conv3t = tt.network.conv2d_transpose("deconv3", conv2t,
                                         FRAME_CHANNELS, 9, 9, 2, 2,
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA))
    tt.board.activation_summary(conv3t)
        
    return conv3t


def conv2d_lstm(conv_image_sequence):
    LSTM_LAYERS = 1
    LSTM_KSIZE = 7
    LSTM_FILTERS = 128
    LSTM_HEIGHT = 30
    LSTM_WIDTH = 40
    
    lstm_cell = tt.recurrent.BasicLSTMConv2DCell(LSTM_KSIZE, LSTM_KSIZE,
                                                 LSTM_FILTERS, LSTM_HEIGHT, LSTM_WIDTH)
    if LSTM_LAYERS > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * LSTM_LAYERS)

    # Get lstm cell output
    outputs, states = tt.recurrent.rnn_conv2d(lstm_cell, conv_image_sequence)
    return outputs[-1]    
    

def inference(stacked_input, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    batch_size = tf.shape(stacked_input)[0]
    static_input_shape = stacked_input.get_shape().as_list()
    input_seq_length = static_input_shape[3] / FRAME_CHANNELS
    input_splitted = tf.split(3, input_seq_length, stacked_input)
    
    # encoder
    with tf.variable_scope("encoder"):
        convolved_inputs = []
        for i, input_part in enumerate(input_splitted):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            convolved_input = encoder(input_part, LAMBDA)
            convolved_inputs.append(convolved_input)
    
    # lstm
    with tf.variable_scope('lstm'):
        lstm_output = conv2d_lstm(convolved_inputs)
    
    # decoder
    with tf.variable_scope('decoder'):
        prediction = decoder(lstm_output, FRAME_CHANNELS, LAMBDA)
    return prediction

    
def loss(model_output, next_frame):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        model_output: Output from inference() function.
        next_frame: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average euc-loss across the batch
    loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model_output, next_frame), 2)), name="euc_loss")
    #loss = tf.contrib.losses.sum_of_pairwise_squares(model_output, next_frame)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="reg_loss")
    
    total_loss = tf.add(loss, reg_loss, name="total_loss")
    return total_loss, loss