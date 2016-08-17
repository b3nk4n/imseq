import tensortools as tt
import tensorflow as tf


def _create_lstm_cell(height, width, layers, filters, ksize_input, ksize_hidden):
    """Helper method to create a LSTMConv2D cell with multiple layers with
       variables bound to CPU memory.
    """
    lstm_cell = tt.recurrent.BasicLSTMConv2DCell(height, width, filters,
                                                 ksize_input=ksize_input,
                                                 ksize_hidden=ksize_hidden,
                                                 device='/cpu:0')
    if layers > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * layers)
    return lstm_cell


def _conv_stack(x, reg_lambda):
    conv1 = tt.network.conv2d("Conv1", x, 32,
                              (3, 3), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu,
                              device='/cpu:0')
    conv2 = tt.network.conv2d("Conv2", conv1, 64,
                              (3, 3), (1, 1),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu,
                              device='/cpu:0')
    conv3 = tt.network.conv2d("Conv3", conv2, 64,
                              (3, 3), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                              activation=tf.nn.relu,
                              device='/cpu:0')
    return conv1, conv2, conv3


def _deconv_stack(representation, conv1, conv2, reg_lambda, channels):
    deconv1 = tt.network.conv2d_transpose("Deconv1", representation, 64,
                                          (3, 3), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          activation=tf.nn.relu,
                                          device='/cpu:0')
    x = tf.concat(3, [deconv1, conv2])
    deconv2 = tt.network.conv2d_transpose("Deconv2", x, 32,
                                          (3, 3), (1, 1),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          activation=tf.nn.relu,
                                          device='/cpu:0')
    x = tf.concat(3, [deconv2, conv1])
    deconv3 = tt.network.conv2d_transpose("Deconv3", x, channels,
                                          (3, 3), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          device='/cpu:0')
    return deconv1, deconv2, deconv3


class ConvLSTMConv2DDecoderEncoderModel(tt.model.AbstractModel):
    """Model that uses the LSTM decoder/encoder architecture
       with LSTMConv2D instead of normal LSTM cells that processes
       the convoluted representation. 
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4,
                 lstm_layers=1, lstm_ksize_input=(7, 7), lstm_ksize_hidden=(7,7)):
        self._lstm_layers = lstm_layers
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        super(ConvLSTMConv2DDecoderEncoderModel, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        input_seq = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, self.input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
        
        with tf.variable_scope("ConvStack") as varscope:
            conv_input_seq = []
            for i in xrange(len(input_seq)):
                if i > 0:
                    varscope.reuse_variables()

                # we only need the very last c1,c2. Override the previous
                (c1, c2, c3) = _conv_stack(input_seq[i], self.reg_lambda)
                conv_input_seq.append(c3)
            
        representation_shape = c3.get_shape().as_list()

        with tf.variable_scope("encoder-lstm"):
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden)
            _, enc_state = tt.recurrent.rnn_conv2d(lstm_cell, conv_input_seq, initial_state=None)  

        # on test/eval: use previously generated frame
        # TODO: use train-code (above) using a longer input-seq (plus a constr-param, that tells the input_length)?
        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden)
            
            rep_outputs, _ = tt.recurrent.rnn_conv2d_roundabout(lstm_cell, conv_input_seq[-1],
                                                                sequence_length=self.output_shape[1],
                                                                initial_state=enc_state)
            
        with tf.variable_scope("DeconvStack") as varscope:
            deconv_output_seq = []
            inputs = (c1, c2)
            for i in xrange(len(rep_outputs)):
                if i > 0:
                    varscope.reuse_variables()
                    
                # we only need the very last c1,c2. Override the previous
                (dc1, dc2, dc3) = _deconv_stack(rep_outputs[i], inputs[0], inputs[1],
                                                self.reg_lambda, self.input_shape[4])
                deconv_output_seq.append(dc3)
                inputs = (dc2, dc1)

        return tf.pack(deconv_output_seq, axis=1)

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
