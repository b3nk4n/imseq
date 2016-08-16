import os
import sys
# add path to libraries for ipython
sys.path.append(os.path.expanduser("~/libs"))
import tensortools as tt

import tensorflow as tf


def encoder(frame_input, LAMBDA):  
    # conv1  
    conv1 = tt.network.conv2d("conv1", frame_input,
                              32, (10, 10), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv1)
    
    # conv2  
    conv2 = tt.network.conv2d("conv2", conv1,
                              64, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv2)
    
    # conv3  
    conv3 = tt.network.conv2d("conv3", conv2,
                              96, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.tanh) # a paper proposes to use TANH here
    tt.board.activation_summary(conv3)
        
    return conv3


def decoder(rep_input, FRAME_CHANNELS, LAMBDA):
    conv1t = tt.network.conv2d_transpose("deconv1", rep_input,
                                         64, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv1t)
    
    conv2t = tt.network.conv2d_transpose("deconv2", conv1t,
                                         32, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv2t)
    
    conv3t = tt.network.conv2d_transpose("deconv3", conv2t,
                                         FRAME_CHANNELS, (10, 10), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(LAMBDA))
    tt.board.activation_summary(conv3t)
        
    return conv3t


def inference(stacked_input, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    batch_size = tf.shape(stacked_input)[0]
    static_input_shape = stacked_input.get_shape().as_list()
    LSTM_SIZE = static_input_shape[1] * static_input_shape[2] * 96 // 4 // 4 // 4
    LSTM_LAYERS = 1
    
    # LSTM-Encoder:
    with tf.variable_scope('lstm'):
        lstm = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
        if LSTM_SIZE > 1:
            multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * LSTM_LAYERS)
        else:
            multi_lstm = lstm
        initial_state = state = multi_lstm.zero_state(batch_size, tf.float32)
        for t_step in xrange(INPUT_SEQ_LENGTH):
            if t_step > 0:
                tf.get_variable_scope().reuse_variables()

            conv_output = encoder(stacked_input[:, :, :, (t_step * FRAME_CHANNELS):(t_step + 1) * FRAME_CHANNELS], 
                                  LAMBDA)
                
            # state value is updated after processing each batch of sequences
            shape_before_lstm = tf.shape(conv_output)
            conv_output_reshaped = tf.reshape(conv_output, [-1, LSTM_SIZE])
            output, state = multi_lstm(conv_output_reshaped, state)
    # exit of reuse-variable scope
    learned_representation = state
    output_representation = tf.reshape(output, shape_before_lstm)

    prediction = decoder(output_representation, FRAME_CHANNELS, LAMBDA)
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
    loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            model_output, next_frame), 2)), name="euc_loss")
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="reg_loss")
    
    total_loss = tf.add(loss, reg_loss, name="total_loss")
    return total_loss, loss