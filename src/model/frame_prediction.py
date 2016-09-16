import tensortools as tt
import tensorflow as tf


def _create_lstm_cell(height, width, layers, filters, ksize_input, ksize_hidden,
                      is_training, memory_device):
    """Helper method to create a LSTMConv2D cell with multiple layers with
       variables bound to CPU memory.
    """
    lstm_cell = tt.recurrent.LSTMConv2DCell(height, width, filters,
                                            ksize_input=ksize_input,
                                            ksize_hidden=ksize_hidden,
                                            use_peepholes=True,
                                            #bn_input_hidden=True,
                                            #bn_hidden_hidden=True,
                                            #bn_peepholes=True,
                                            #is_training=is_training,
                                            device=memory_device)
    if layers > 1:
        lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * layers)
    return lstm_cell


def _conv_stack(x, weight_decay, is_training, memory_device):
    #x = tf.contrib.layers.batch_norm(x, is_training=is_training, scope="x_bn")
    conv1 = tt.network.conv2d("Conv1", x, 32,
                              (5, 5), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                              activation=tf.nn.relu,
                              device=memory_device)

    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope="conv1_bn")
    conv2 = tt.network.conv2d("Conv2", conv1, 64,
                              (3, 3), (1, 1),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                              activation=tf.nn.relu,
                              device=memory_device)

    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope="conv2_bn")
    conv3 = tt.network.conv2d("Conv3", conv2, 64,
                              (3, 3), (2, 2),
                              weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_init=0.1,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                              activation=tf.nn.relu,
                              device=memory_device)

    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training, scope="conv3_bn")
    return conv3


def _deconv_stack(representation, weight_decay, channels, is_training, memory_device):
    representation = tf.contrib.layers.batch_norm(representation,
                                                  is_training=is_training,
                                                  scope="rep_bn")
    deconv1 = tt.network.conv2d_transpose("Deconv1", representation, 64,
                                          (3, 3), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                          activation=tf.nn.relu,
                                          device=memory_device)

    deconv1 = tf.contrib.layers.batch_norm(deconv1, is_training=is_training, scope="deconv1_bn")
    deconv2 = tt.network.conv2d_transpose("Deconv2", deconv1, 32,
                                          (3, 3), (1, 1),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                          activation=tf.nn.relu,
                                          device=memory_device)

    deconv2 = tf.contrib.layers.batch_norm(deconv2, is_training=is_training, scope="deconv2_bn")
    deconv3 = tt.network.conv2d_transpose("Deconv3", deconv2, channels,
                                          (5, 5), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                          activation=tf.nn.sigmoid,
                                          device=memory_device)
                                          # Note: No-Non-lin. give best results for MovingMNIST
                                          # when no BN is used!
    return deconv3


class LSTMConv2DPredictionModel(tt.model.AbstractModel):    
    def __init__(self, weight_decay,
                 lstm_layers=2, lstm_ksize_input=(5, 5), lstm_ksize_hidden=(5,5)):
        self._lstm_layers = lstm_layers
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        super(LSTMConv2DPredictionModel, self).__init__(weight_decay)
        
    @tt.utils.attr.override
    def inference(self, inputs, targets, feeds,
                  is_training, device_scope, memory_device):
        input_shape = inputs.get_shape().as_list()
        target_shape = targets.get_shape().as_list()
        
        input_seq = tf.transpose(inputs, [1, 0, 2, 3, 4])
        input_seq = tf.split(0, input_shape[1], input_seq)
        input_seq = [tf.squeeze(i, (0,)) for i in input_seq]
        
        with tf.variable_scope("ConvStack") as varscope:
            conv_input_seq = []
            for i in xrange(len(input_seq)):
                if i > 0:
                    varscope.reuse_variables()
                
                c3 = _conv_stack(input_seq[i], self.weight_decay,
                                 is_training, memory_device)
                conv_input_seq.append(c3)
            
        representation_shape = c3.get_shape().as_list()

        with tf.variable_scope("encoder-lstm"):
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden,
                                          is_training, memory_device)
            _, enc_state = tt.recurrent.rnn_conv2d(lstm_cell, conv_input_seq, initial_state=None)  

        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden,
                                          is_training, memory_device)
            
            rep_outputs, _ = tt.recurrent.rnn_conv2d_roundabout(lstm_cell, conv_input_seq[-1],
                                                                sequence_length=target_shape[1],
                                                                initial_state=enc_state)
            
        with tf.variable_scope("DeconvStack") as varscope:
            deconv_output_seq = []
            for i in xrange(len(rep_outputs)):
                if i > 0:
                    varscope.reuse_variables()

                dc3 = _deconv_stack(rep_outputs[i], self.weight_decay, input_shape[4],
                                    is_training, memory_device)
                deconv_output_seq.append(dc3)

        packed_result = tf.pack(deconv_output_seq, axis=1)
        
        return packed_result
    
    @tt.utils.attr.override
    def loss(self, predictions, targets, device_scope):
        # BCE loss
        bce_loss = tt.loss.bce(predictions, targets)
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_loss)
        
        # mGDL loss
        pshape = predictions.get_shape().as_list()
        tshape = targets.get_shape().as_list()
        pred_reshaped = tf.reshape(predictions, [-1] + pshape[2:])
        tgt_reshaped = tf.reshape(targets, [-1] + tshape[2:])
        gdl_loss = tt.loss.mgdl(pred_reshaped, tgt_reshaped)
        tf.add_to_collection(tt.core.LOG_LOSSES, gdl_loss)
        
        # additional single-frame losses to eval training over time
        bce_fr1 = tt.loss.bce(predictions[:,0,:,:,:], targets[:,0,:,:,:], name="1st_frame")
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_fr1)
        bce_fr2 = tt.loss.bce(predictions[:,1,:,:,:], targets[:,1,:,:,:], name="2nd_frame")
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_fr2)
        bce_fr3 = tt.loss.bce(predictions[:,2,:,:,:], targets[:,2,:,:,:], name="3rd_frame")
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_fr3)
        bce_last = tt.loss.bce(predictions[:,pshape[1],:,:,:], targets[:,tshape[1],:,:,:], name="last_frame")
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_last)
        
        # combine both losses to optimize
        return tf.add(bce_loss, gdl_loss, name="combined_loss")
    
    @tt.utils.attr.override
    def evaluation(self, predictions, targets, device_scope):
        # swap batch_size and time-dim
        swapped_predictions = tf.transpose(predictions, [1,0,2,3,4])
        swapped_targets = tf.transpose(targets, [1,0,2,3,4])

        # split as list of element-shape [bs, h, w, c]
        predictions_list = tf.unpack(swapped_predictions)
        targets_list = tf.unpack(swapped_targets)
        
        psnr = sharpdiff = ssim = 0.0
        for pred, tgt in zip(predictions_list, targets_list):
            psnr += tt.image.psnr(pred, tgt)
            sharpdiff += tt.image.sharp_diff(pred, tgt)
            ssim += tt.image.ssim(pred, tgt)
            
        # average values
        denom = len(targets_list)
        psnr /= denom
        sharpdiff /= denom
        ssim /= denom
            
        # add bce/mse losses here as well to see them in the evaluation
        bce = tt.loss.bce(predictions, targets)
        mse = tt.loss.mse(predictions, targets)

        return {"bce": bce, "mse": mse, "psnr": psnr, "sharpdiff": sharpdiff, "ssim": ssim}