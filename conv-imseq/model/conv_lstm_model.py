import tensortools as tt
import tensorflow as tf


def _create_lstm_cell(size, layers):
    """Helper method to create a LSTM cell with multiple layers with
       variables bound to CPU memory.
    """
    lstm_cell = tf.recurrent.BasicLSTMCell(size,
                                           state_is_tuple=True,
                                           device='/cpu:0')
    if layers > 1:
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * LSTM_LAYERS)
    return lstm_cell


def _encoder(frame_input, reg_lambda): 
    """Helper to create a convolutional encoder."""
    # conv1  
    conv1 = tt.network.conv2d("conv1", frame_input,
                              32, (10, 10), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv1)
    
    conv2 = tt.network.conv2d("conv2", conv1,
                              64, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu)
    tt.board.activation_summary(conv2)
    
    conv3 = tt.network.conv2d("conv3", conv2,
                              96, (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.tanh) # a paper proposes to use TANH here
    tt.board.activation_summary(conv3)
        
    return conv3


def _decoder(rep_input, reg_lambda, channels):
    """Helper to create a convolutional decoder."""
    conv1t = tt.network.conv2d_transpose("deconv1", rep_input,
                                         64, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv1t)
    
    conv2t = tt.network.conv2d_transpose("deconv2", conv1t,
                                         32, (5, 5), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                         activation=tf.nn.relu)
    tt.board.activation_summary(conv2t)
    
    conv3t = tt.network.conv2d_transpose("deconv3", conv2t,
                                         channels, (10, 10), (2, 2),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))
    tt.board.activation_summary(conv3t)
        
    return conv3t


class ConvLSTMModel(tt.model.AbstractModel):
    """Model that uses a convolutional autoencoder and and a LSTM in-between
       to model the temporal dimension.
       This model only predicts a single frame.
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4,
                 lstm_layers=1, lstm_size=2048):
        self._lstm_layers = lstm_layers
        self._lstm_size = lstm_size
        super(ConvLSTMModel, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        input_seq = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, self.input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
        stacked_input = tf.concat(3, input_seq)
        
        frame_channels = self.input_shape[4]
        
        # LSTM-Encoder:
        with tf.variable_scope('lstm'):
            lstm_cell = _create_lstm_cell(64*64//8, self._lstm_layers)
            initial_state = state = lstm_cell.zero_state(self.batch_size, tf.float32)
            for t_step in xrange(self.input_shape[1]):
                if t_step > 0:
                    tf.get_variable_scope().reuse_variables()

                conv_output = _encoder(stacked_input[:, :, :, (t_step * frame_channels):(t_step + 1) * frame_channels], 
                                      self.reg_lambda)

                # state value is updated after processing each batch of sequences
                shape_before_lstm = tf.shape(conv_output)
                conv_output_reshaped = tf.reshape(conv_output, [self.batch_size, 64*64//8])
                output, state = lstm_cell(conv_output_reshaped, state)
        # exit of reuse-variable scope
        learned_representation = state
        output_representation = tf.reshape(output, shape_before_lstm)

        prediction = _decoder(output_representation, self.reg_lambda, frame_channels)
        prediction = tf.expand_dims(prediction, 1)
        return prediction

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
