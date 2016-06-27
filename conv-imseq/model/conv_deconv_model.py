import os
import sys
# add path to libraries for ipython
sys.path.append(os.path.expanduser("~/libs"))
import tensortools as tt

import tensorflow as tf


def inference(stacked_input, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    # conv1  
    conv1 = tt.network.conv2d("conv1", stacked_input,
                              64, 5, 5, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    
    # conv2  
    conv2 = tt.network.conv2d("conv2", conv1,
                              128, 5, 5, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    
    # conv3       
    conv3 = tt.network.conv2d("conv3", conv2,
                              256, 5, 5, 2, 2,
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                              activation=tf.nn.relu)
    
    # conv_tp4
    conv_tp4 = tt.network.conv2d_transpose("conv_tp4", conv3,
                                           128, 5, 5, 2, 2,
                                           weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                           bias=0.1,
                                           regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                           activation=tf.nn.relu)
    
    # conv_tp5  
    conv_tp5 = tt.network.conv2d_transpose("conv_tp5", conv_tp4,
                                           64, 5, 5, 2, 2,
                                           weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                           bias=0.1,
                                           regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                           activation=tf.nn.relu)
    
    # conv_tp6       
    conv_tp6 = tt.network.conv2d_transpose("conv_tp6", conv_tp5,
                                           FRAME_CHANNELS, 5, 5, 2, 2,
                                           weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                           bias=0.1,
                                           regularizer=tf.contrib.layers.l2_regularizer(LAMBDA),
                                           activation=tf.nn.relu)
        
    return conv_tp6


def loss(model_output, next_frame):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average L2 loss across the batch
    euc_loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            model_output, next_frame), 2)))
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    total_loss = euc_loss + reg_loss
    return total_loss