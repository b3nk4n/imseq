import os
import sys
# add path to libraries for ipython
sys.path.append(os.path.expanduser("~/libs"))
import tensortools as tt

import tensorflow as tf


def inference(stacked_input, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    with tf.name_scope("encoder"):
        # conv1  
        conv1 = tt.network.conv2d("conv1", stacked_input,
                                  64, (5, 5), (2, 2),
                                  weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                  bias_init=0.1,
                                  regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                  activation=tf.nn.relu)
        tt.board.activation_summary(conv1)
        tt.board.conv_image_summary("conv1_out", conv1)

        # conv2  
        conv2 = tt.network.conv2d("conv2", conv1,
                                  128, (5, 5), (2, 2),
                                  weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                  bias_init=0.1,
                                  regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                  activation=tf.nn.relu)
        tt.board.activation_summary(conv2)
        tt.board.conv_image_summary("conv2_out", conv2)

        # conv3       
        conv3 = tt.network.conv2d("conv3", conv2,
                                  256, (5, 5), (2, 2),
                                  weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                  bias_init=0.1,
                                  regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                  activation=tf.nn.relu)
        tt.board.activation_summary(conv3)
        tt.board.conv_image_summary("conv3_out", conv3)
    
    with tf.name_scope("decoder"):
        # conv_tp4
        conv_tp4 = tt.network.conv2d_transpose("conv_tp4", conv3,
                                               128, (5, 5), (2, 2),
                                               weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                               bias_init=0.1,
                                               regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                               activation=tf.nn.relu)
        tt.board.activation_summary(conv_tp4)

        # conv_tp5  
        conv_tp5 = tt.network.conv2d_transpose("conv_tp5", conv_tp4,
                                               64, (5, 5), (2, 2),
                                               weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                               bias_init=0.1,
                                               regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                               activation=tf.nn.relu)
        tt.board.activation_summary(conv_tp5)

        # conv_tp6       
        conv_tp6 = tt.network.conv2d_transpose("conv_tp6", conv_tp5,
                                               FRAME_CHANNELS, (5, 5), (2, 2),
                                               weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                               bias_init=0.1,
                                               regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                               activation=tf.nn.relu)
        tt.board.activation_summary(conv_tp6)
        
    return conv_tp6


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