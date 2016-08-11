import tensortools as tt
import tensorflow as tf


def _create_lstm_cell():
    LSTM_LAYERS = 1
    LSTM_KSIZE = 7
    LSTM_FILTERS = 128
    LSTM_HEIGHT = 64
    LSTM_WIDTH = 64
    lstm_cell = tt.recurrent.BasicLSTMConv2DCell(LSTM_KSIZE, LSTM_KSIZE,
                                                 LSTM_FILTERS, LSTM_HEIGHT, LSTM_WIDTH,
                                                 device='/cpu:0')
    if LSTM_LAYERS > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * LSTM_LAYERS)
    return lstm_cell
    

def _conv2d_lstm(image_list, initial_state):
    lstm_cell = _create_lstm_cell()
        
    # Get lstm cell output
    outputs, states = tt.recurrent.rnn_conv2d(lstm_cell, image_list, initial_state=initial_state)
    return outputs, states      


def inference(input_seq, pred_seq, FRAME_CHANNELS, OUTPUT_SEQ_LENGTH, LAMBDA): # TODO: INPUT => OUTPUT SEQ LENGTH
    # input_seq: [batch_size, n_steps, h, w, c]
    batch_size = tf.shape(input_seq)[0]
    static_input_shape = input_seq.get_shape().as_list()
    input_seq = tf.transpose(input_seq, [1, 0, 2, 3, 4])
    input_seq = tf.split(0, static_input_shape[1], input_seq)
    input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
    
    with tf.variable_scope("encoder-lstm"):
        enc_outputs, enc_state = _conv2d_lstm(input_seq, None)
    
    if pred_seq is not None:
        # on training:  use ground truth previous frames 
        static_pred_shape = pred_seq.get_shape().as_list()
        pred_seq = tf.transpose(pred_seq, [1, 0, 2, 3, 4])
        pred_seq = tf.split(0, static_pred_shape[1], pred_seq)
        pred_seq_list = [input_seq[-1]]
        for i in xrange(static_pred_shape[1] - 1):
            pred_seq_list.append(tf.squeeze(pred_seq[i], (0,)))

        with tf.variable_scope('decoder-lstm'):
            dec_outputs, _ = _conv2d_lstm(pred_seq_list, enc_state)

            for i in xrange(len(dec_outputs)):
                dec_outputs[i] = tt.network.conv2d("Conv-Reduce", dec_outputs[i], 1,
                                                   3, 3, 1, 1,
                                                   weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                   bias_init=0.1,
                                                   regularizer=tf.contrib.layers.l2_regularizer(LAMBDA))
                tf.get_variable_scope().reuse_variables()

            return tf.pack(dec_outputs, axis=1)
        
    else:
        # on test/eval: use previously generated frame
        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell()
            input_ = input_seq[-1]
            state = enc_state
            dec_outputs = []
            
            for time in xrange(OUTPUT_SEQ_LENGTH):
                if time > 0:
                        varscope.reuse_variables()
                        
                with tf.variable_scope("RNNConv2D"): 
                    (output, state) = lstm_cell(input_, state)

                dec_output = tt.network.conv2d("Conv-Reduce", output, 1,
                                               3, 3, 1, 1,
                                               weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                               bias_init=0.1,
                                               regularizer=tf.contrib.layers.l2_regularizer(LAMBDA))
                _input = dec_output
                dec_outputs.append(dec_output)

            return tf.pack(dec_outputs, axis=1)

    
def loss(model_outputs, next_frames):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        model_output: Output from inference() function.
        next_frames: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average squared loss per image across the batch
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.pow(tf.sub(model_outputs, next_frames), 2), reduction_indices=[2,3,4]),
        name="squared_loss")
    
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="reg_loss")
    
    total_loss = tf.add(loss, reg_loss, name="total_loss")
    return total_loss, loss