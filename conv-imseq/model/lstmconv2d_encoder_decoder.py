import tensortools as tt
import tensorflow as tf


def _create_lstm_cell(height, width, layers, filters, ksize_input, ksize_hidden, memory_device):
    """Helper method to create a LSTMConv2D cell with multiple layers with
       variables bound to CPU memory.
    """
    lstm_cell = tt.recurrent.BasicLSTMConv2DCell(height, width, filters,
                                                 ksize_input=ksize_input,
                                                 ksize_hidden=ksize_hidden,
                                                 device=memory_device)
    if layers > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * layers)
    return lstm_cell


class LSTMConv2DDecoderEncoderModel(tt.model.AbstractModel): 
    """Model that uses the LSTM decoder/encoder architecture,
       but with LSTMConv2D instead of normal LSTM cells.
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, reg_lambda=0.0,
                 lstm_layers=1, lstm_filters=64, lstm_ksize_input=(7, 7), lstm_ksize_hidden=(7,7)):
        self._lstm_layers = lstm_layers
        self._lstm_filters = lstm_filters
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        super(LSTMConv2DDecoderEncoderModel, self).__init__(reg_lambda)
        
    @tt.utils.attr.override
    def inference(self, inputs, targets, feeds,
                  is_training, device_scope, memory_device):
        input_shape = inputs.get_shape().as_list()
        target_shape = targets.get_shape().as_list()
        
        # rescale values from [0, 1] to [-1, 1]
        inputs = inputs * 2.0 - 1.0
        
        input_seq = tf.transpose(inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]

        with tf.variable_scope("encoder-lstm"):
            lstm_cell = _create_lstm_cell(input_shape[2], input_shape[3],
                                          self._lstm_layers, self._lstm_filters, 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden,
                                          memory_device)
            _, enc_state = tt.recurrent.rnn_conv2d(lstm_cell, input_seq, initial_state=None)  

        '''if pred_seq is not None:
            # on training:  use ground truth previous frames 
            static_pred_shape = pred_seq.get_shape().as_list()
            pred_seq = tf.transpose(pred_seq, [1, 0, 2, 3, 4])
            pred_seq = tf.split(0, static_pred_shape[1], pred_seq)
            pred_seq_list = [input_seq[-1]]
            for i in xrange(static_pred_shape[1] - 1):
                pred_seq_list.append(tf.squeeze(pred_seq[i], (0,)))

            with tf.variable_scope('decoder-lstm'):
                lstm_cell = _create_lstm_cell(input_shape[2], input_shape[3],
                                              self._lstm_layers, self._lstm_filters, 
                                              self._lstm_ksize_input, self._lstm_ksize_hidden)
                dec_outputs, _ = tt.recurrent.rnn_conv2d(lstm_cell, pred_seq_list, initial_state=enc_state)

                for i in xrange(len(dec_outputs)):
                    dec_outputs[i] = tt.network.conv2d("Conv-Reduce", dec_outputs[i], 1,
                                                       (1, 1), (1, 1),
                                                       weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       bias_init=0.1,
                                                       regularizer=tf.contrib.layers.l2_regularizer(self._reg_lambda))
                    tf.get_variable_scope().reuse_variables()

                return tf.pack(dec_outputs, axis=1)

        else:'''
        # on test/eval: use previously generated frame
        # TODO: use train-code (above) using a longer input-seq (plus a constr-param, that tells the input_length)?
        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell(input_shape[2], input_shape[3],
                                          self._lstm_layers, self._lstm_filters, 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden,
                                          memory_device)
            
            def postprocessor(x):
                return tt.network.conv2d("Conv-Reduce", x, input_shape[4],
                                         (3, 3), (1, 1),
                                         weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                         bias_init=0.1,
                                         regularizer=tf.contrib.layers.l2_regularizer(self.reg_lambda),
                                         activation=tf.nn.tanh,
                                         device=memory_device)
            
            dec_outputs, _ = tt.recurrent.rnn_conv2d_roundabout(lstm_cell, input_seq[-1],
                                                                sequence_length=target_shape[1],
                                                                initial_state=enc_state,
                                                                output_postprocessor=postprocessor)

            packed_result = tf.pack(dec_outputs, axis=1)
            packed_result.set_shape(target_shape)

            # rescale values from [-1, 1] to [0, 1]
            packed_result = (packed_result + 1.0) / 2.0
        
            return packed_result       

    
    @tt.utils.attr.override
    def loss(self, predictions, targets, device_scope):
        return tt.loss.bce(predictions, targets)