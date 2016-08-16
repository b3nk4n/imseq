import tensortools as tt
import tensorflow as tf


import tensortools as tt
import tensorflow as tf


def _create_lstm_cell(layers, size):
    """Helper method to create a LSTM cell with multiple layers with
       variables bound to CPU memory.
    """
    lstm_cell = tt.recurrent.BasicLSTMCell(size,
                                           state_is_tuple=True,
                                           device='/cpu:0')
    if layers > 1:
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layers)
    return lstm_cell


class LSTMDecoderEncoderModel(tt.model.AbstractModel):
    """Model that uses the LSTM decoder/encoder architecture,
       but with LSTMConv2D instead of normal LSTM cells.
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4,
                 lstm_layers=1):
        self._lstm_layers = lstm_layers
        super(LSTMDecoderEncoderModel, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        input_seq = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, self.input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
        
        # flatten (3D->1D)
        for i in xrange(len(input_seq)):
            input_seq[i] = tf.contrib.layers.flatten(input_seq[i])

        with tf.variable_scope("encoder-lstm"):
            lstm_cell = _create_lstm_cell(self._lstm_layers,
                                          self.input_shape[2] * self.input_shape[3])
            _, enc_state = tf.nn.rnn(lstm_cell, input_seq, initial_state=None, dtype=tf.float32)  

        
        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell(self._lstm_layers,
                                          self.input_shape[2] * self.input_shape[3])
            input_ = input_seq[-1]
            state = enc_state
            dec_outputs = []

            for t in xrange(self._targets.get_shape()[1]):
                if t > 0:
                    varscope.reuse_variables()

                with tf.variable_scope("RNN"): 
                    (output, state) = lstm_cell(input_, state)

                dec_output = tf.reshape(output, [-1, self.input_shape[2], self.input_shape[3], self.input_shape[4]])
                _input = dec_output
                dec_outputs.append(dec_output)

            return tf.pack(dec_outputs, axis=1)

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
