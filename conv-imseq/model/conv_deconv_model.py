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


def inference(stacked_input, BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, INPUT_SEQ_LENGTH, LAMBDA):
    # Shape (8, 2, 240, 320, 3)
    # TODO: this preprocessing step could be moved to inputs()
    # input_frames = []
    # for i in xrange(INPUT_SEQ_LENGTH):
    #     input_frames.append(input_seq[:,i,:,:,:])
    # stacked_input = tf.concat(3, input_frames)
    
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, FRAME_CHANNELS * INPUT_SEQ_LENGTH, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(stacked_input, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv1)
        
    # norm1
    # norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 128],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv2)
        
    # norm2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 128, 256],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv3)
        
    # norm3
    # norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    
    # conv_tp4
    with tf.variable_scope('conv_tp4') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 128, 256],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv3, kernel, 
                                      [BATCH_SIZE, FRAME_HEIGHT // 4, FRAME_WIDTH // 4, 128],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv_tp4 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv_tp4)
        
    # norm4
    # norm4 = tf.nn.lrn(conv_tp4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    
    # conv_tp5
    with tf.variable_scope('conv_tp5') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 64, 128],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv_tp4, kernel,
                                      [BATCH_SIZE, FRAME_HEIGHT // 2, FRAME_WIDTH // 2, 64],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv_tp5 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv_tp5)
        
    # norm5
    # norm5 = tf.nn.lrn(conv_tp5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')
    
    # conv_tp6
    with tf.variable_scope('conv_tp6') as scope:
        kernel = variable_with_wd('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=LAMBDA)
        conv = tf.nn.conv2d_transpose(conv_tp5, kernel,
                                      [BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                                      [1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [3], initializer=tf.constant_initializer(0.0))
        conv_tp6 = tf.nn.bias_add(conv, biases)
        
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
    # Calculate the average L2 loss across the batch.
    # squeezed_next_frame = tf.squeeze(next_frame)
    # l2loss = tf.nn.l2_loss(model_output - squeezed_next_frame)
    l2loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            model_output, next_frame), 2)))
    tf.add_to_collection('losses', l2loss)

    # The total loss is defined as the L2 loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_wd')