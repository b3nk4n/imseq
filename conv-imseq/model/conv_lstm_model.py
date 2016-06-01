import tensorflow as tf

def variable_with_wd(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape,
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def convolutions(frame_input, BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, LAMBDA):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, FRAME_CHANNELS, 32],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(frame_input, kernel, [1, 5, 5, 1], padding='SAME')
        biases = tf.get_variable('biases', [32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
         
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 32, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        
    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        
    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 96],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(conv3, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name) # TODO: remove this relu here, before passing it to LSTM?
        
    return conv4


def convolutions_transposed(rep_input, BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, LAMBDA):
    # conv_tp4
    with tf.variable_scope('conv_tp4') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 96],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(rep_input, kernel, 
                                      [BATCH_SIZE, FRAME_HEIGHT // 20, FRAME_WIDTH // 20, 64],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv_tp4 = tf.nn.relu(bias, name=scope.name)
    
    # conv_tp5
    with tf.variable_scope('conv_tp5') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv_tp4, kernel,
                                      [BATCH_SIZE, FRAME_HEIGHT // 10, FRAME_WIDTH // 10, 64],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv_tp5 = tf.nn.relu(bias, name=scope.name)
    
    # conv_tp6
    with tf.variable_scope('conv_tp6') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 32, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv_tp5, kernel,
                                      [BATCH_SIZE, FRAME_HEIGHT // 5, FRAME_WIDTH // 5, 32],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [32], initializer=tf.constant_initializer(0.0))
        conv_tp6 = tf.nn.relu(tf.nn.bias_add(conv, biases))

    # conv_tp7
    with tf.variable_scope('conv_tp7') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 3, 32],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv_tp6, kernel,
                                      [BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                                      [1, 5, 5, 1], padding='SAME')
        biases = tf.get_variable('biases', [3], initializer=tf.constant_initializer(0.0))
        conv_tp7 = tf.nn.bias_add(conv, biases) # no ReLu here
        
    return conv_tp7


def inference(stacked_input, BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    LSTM_SIZE = FRAME_HEIGHT * FRAME_WIDTH * 96 // 4 // 4 // 4 // 25
    LSTM_LAYERS = 1
    
    # LSTM-Encoder:
    with tf.variable_scope('LSTM'):
        lstm = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
        multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * LSTM_LAYERS)
        initial_state = state = multi_lstm.zero_state(BATCH_SIZE, tf.float32)
        for t_step in xrange(INPUT_SEQ_LENGTH):
            if t_step > 0:
                tf.get_variable_scope().reuse_variables()

            conv_output = convolutions(stacked_input[:, :, :, (t_step * FRAME_CHANNELS):(t_step + 1) * FRAME_CHANNELS], 
                                       BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, LAMBDA)
                
            # state value is updated after processing each batch of sequences
            shape_before_lstm = tf.shape(conv_output)
            conv_output_reshaped = tf.reshape(conv_output, [BATCH_SIZE, -1])
            output, state = multi_lstm(conv_output_reshaped, state)
    # exit of reuse-variable scope
           
    learned_representation = state
    output_representation = tf.reshape(output, shape_before_lstm)

    prediction = convolutions_transposed(output_representation, BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, LAMBDA)

    return prediction

    
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
    # Calculate the average L2 loss across the batch.
    # squeezed_next_frame = tf.squeeze(next_frame)
    # l2loss = tf.nn.l2_loss(model_output - squeezed_next_frame)
    # l2loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
    #         model_output, next_frame), 2)))
    mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(model_output - next_frame), reduction_indices=[1, 2, 3]))
    tf.add_to_collection('losses', mse_loss)

    # The total loss is defined as the L2 loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_wd')