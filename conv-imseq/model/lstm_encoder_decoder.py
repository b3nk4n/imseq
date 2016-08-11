import tensortools as tt
import tensorflow as tf


def _create_lstm_cell():
    LSTM_LAYERS = 1
    LSTM_HIDDEN = 64*64 # other values than 64*64 need a FC-layer at the end!
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN,
                                             state_is_tuple=True)
    if LSTM_LAYERS > 1:
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * LSTM_LAYERS)
    return lstm_cell
    

def _run_lstm(image_list, initial_state):
    lstm_cell = _create_lstm_cell()
        
    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, image_list, initial_state=initial_state, dtype=tf.float32)

    return outputs, states      


def inference(input_seq, pred_seq, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA): # TODO: make INPUT_SEQ_LENGTH ==> OUTPUT_SEQ_LENGT param?
    with tf.device('/cpu:0'):
        # input_seq: [batch_size, n_steps, h, w, c]
        batch_size = tf.shape(input_seq)[0]
        static_input_shape = input_seq.get_shape().as_list()
        input_seq = tf.transpose(input_seq, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, static_input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]

        # flatten (3D->4)
        for i in xrange(len(input_seq)):
            input_seq[i] = tf.contrib.layers.flatten(input_seq[i])

        with tf.variable_scope("encoder-lstm"):
            enc_outputs, enc_state = _run_lstm(input_seq, None)

        # conditional future predictor
        if pred_seq is not None:
            # on training:  use ground truth previous frames 
            static_pred_shape = pred_seq.get_shape().as_list()
            pred_seq = tf.transpose(pred_seq, [1, 0, 2, 3, 4])
            pred_seq = tf.split(0, static_pred_shape[1], pred_seq)
            pred_seq_list = [input_seq[-1]]
            for i in xrange(static_pred_shape[1] - 1):
                pred_seq_list.append(tf.squeeze(pred_seq[i], (0,)))

            # flatten (3D->4)
            for i in xrange(len(pred_seq_list)):
                pred_seq_list[i] = tf.contrib.layers.flatten(pred_seq_list[i])

            with tf.variable_scope('decoder-lstm1'):
                dec_outputs, _ = _run_lstm(pred_seq_list, enc_state)

                for i in xrange(len(dec_outputs)):
                    dec_outputs[i] = tf.reshape(dec_outputs[i], [-1, 64, 64, FRAME_CHANNELS])

            model_outputs = tf.pack(dec_outputs, axis=1)
        else:
            # on test/eval: use previously generated frame
            with tf.variable_scope('decoder-lstm2') as varscope:
                lstm_cell = _create_lstm_cell()
                input_ = input_seq[-1]
                state = enc_state
                dec_outputs = []

                for time in xrange(INPUT_SEQ_LENGTH): # FIXME output seq length
                    if time > 0:
                        varscope.reuse_variables()

                    with tf.variable_scope("RNN"): 
                        (output, state) = lstm_cell(input_, state)

                    dec_output = tf.reshape(output, [-1, 64, 64, FRAME_CHANNELS])
                    _input = dec_output
                    dec_outputs.append(dec_output)

                model_outputs = tf.pack(dec_outputs, axis=1)

        return model_outputs

    
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
    
    #reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="reg_loss")
    
    #total_loss = tf.add(loss, reg_loss, name="total_loss")
    total_loss = tf.identity(loss)
    return total_loss, loss