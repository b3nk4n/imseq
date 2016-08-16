import tensortools as tt
import tensorflow as tf


class ConvDeconvModel(tt.model.AbstractModel):
    """Simple model that uses a convolutional autoencoder architecture.
       It has to learn the temporal dependencies by its own.
       This model only predicts a single frame.
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4):
        super(ConvDeconvModel, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        
        input_seq = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, self.input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
        stacked_input = tf.concat(3, input_seq)
        
        with tf.name_scope("encoder"):
            # conv1  
            conv1 = tt.network.conv2d("conv1", stacked_input,
                                      64, (5, 5), (2, 2),
                                      weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_init=0.1,
                                      regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                      activation=tf.nn.relu)
            tt.board.activation_summary(conv1)
            tt.board.conv_image_summary("conv1_out", conv1)

            # conv2  
            conv2 = tt.network.conv2d("conv2", conv1,
                                      128, (5, 5), (2, 2),
                                      weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_init=0.1,
                                      regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                      activation=tf.nn.relu)
            tt.board.activation_summary(conv2)
            tt.board.conv_image_summary("conv2_out", conv2)

            # conv3       
            conv3 = tt.network.conv2d("conv3", conv2,
                                      256, (5, 5), (2, 2),
                                      weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_init=0.1,
                                      regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                      activation=tf.nn.relu)
            tt.board.activation_summary(conv3)
            tt.board.conv_image_summary("conv3_out", conv3)

        with tf.name_scope("decoder"):
            # conv_tp4
            conv_tp4 = tt.network.conv2d_transpose("conv_tp4", conv3,
                                                   128, (5, 5), (2, 2),
                                                   weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                   bias_init=0.1,
                                                   regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                                   activation=tf.nn.relu)
            tt.board.activation_summary(conv_tp4)

            # conv_tp5  
            conv_tp5 = tt.network.conv2d_transpose("conv_tp5", conv_tp4,
                                                   64, (5, 5), (2, 2),
                                                   weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                   bias_init=0.1,
                                                   regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                                   activation=tf.nn.relu)
            tt.board.activation_summary(conv_tp5)

            # conv_tp6       
            conv_tp6 = tt.network.conv2d_transpose("conv_tp6", conv_tp5,
                                                   self.input_shape[4], (5, 5), (2, 2),
                                                   weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                   bias_init=0.1,
                                                   regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                                   activation=tf.nn.relu)
            tt.board.activation_summary(conv_tp6)

        prediction = tf.expand_dims(conv_tp6, 1)
        return prediction

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
