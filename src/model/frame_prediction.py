import tensortools as tt
import tensorflow as tf


class LSTMConv2DPredictionModel(tt.model.AbstractModel):
    def __init__(self, weight_decay, filters=[32, 64, 64], ksizes=[(5,5),(3,3),(3,3)],
                 strides=[(2,2),(1,1),(2,2)], bias_init=0.1, output_activation=tf.nn.sigmoid,
                 bn_feature_enc=True, bn_feature_dec=True, 
                 lstm_layers=1, lstm_ksize_input=(5, 5), lstm_ksize_hidden=(5,5),
                 lstm_use_peepholes=True, lstm_cell_clip=None, lstm_bn_input_hidden=False, 
                 lstm_bn_hidden_hidden=False, lstm_bn_peepholes=False,
                 scheduled_sampling_decay_rate=None):
        assert len(filters) == len(ksizes) and len(filters) == len(strides), "Encoder/Decoder configuration mismatch."
        
        # feature encoder/decoder
        self._filters = filters
        self._ksizes = ksizes
        self._strides = strides
        self._bias_init = bias_init
        self._output_activation = output_activation
        self._bn_feature_enc = bn_feature_enc
        self._bn_feature_dec = bn_feature_dec
        
        # lstm
        self._lstm_layers = lstm_layers
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        self._lstm_use_peepholes = lstm_use_peepholes
        self._lstm_cell_clip = lstm_cell_clip
        self._lstm_bn_input_hidden = lstm_bn_input_hidden
        self._lstm_bn_hidden_hidden = lstm_bn_hidden_hidden
        self._lstm_bn_peepholes = lstm_bn_peepholes
        
        # scheduled sampling
        self._scheduled_sampling_decay_rate = scheduled_sampling_decay_rate

        super(LSTMConv2DPredictionModel, self).__init__(weight_decay)
        
    @tt.utils.attr.override
    def inference(self, inputs, targets, feeds,
                  is_training, device_scope, memory_device):
        input_shape = inputs.get_shape().as_list()
        target_shape = targets.get_shape().as_list()
        
        # Conv-Encoder
        conv_input_seq = []
        with tf.variable_scope("conv-encoder") as conv_enc_scope:
            # convert from shape [bs, t, h, w, c] to list([bs, h, w, c])
            input_seq = tf.unpack(inputs, axis=1)
            
            for i in xrange(len(input_seq)):
                if i > 0:
                    conv_enc_scope.reuse_variables()
                
                conv = self._conv_stack(input_seq[i], is_training, memory_device)
                conv_input_seq.append(conv)
        
        # shape for convolved inputs that flows through our LSTMs
        feat_repr_shape = conv.get_shape().as_list()[1:]

        # LSTM-Encoder
        with tf.variable_scope("lstm-encoder"):
            lstm_cell = self._create_lstm_cell(feat_repr_shape, is_training, memory_device)
            _, enc_state = tt.recurrent.rnn_conv2d(lstm_cell, conv_input_seq, initial_state=None)  

        # visualilze learned motion pattern
        learned_motion = enc_state
        tt.board.lstm_state_image_summary("motion", learned_motion)
            
        # LSTM-Decoder
        with tf.variable_scope('lstm-decoder') as varscope:
            lstm_cell = self._create_lstm_cell(feat_repr_shape, is_training, memory_device)
            
            if self._scheduled_sampling_decay_rate is None:
                # Always sampling
                rep_outputs, _ = tt.recurrent.rnn_conv2d_roundabout(lstm_cell, conv_input_seq[-1],
                                                                    sequence_length=target_shape[1],
                                                                    initial_state=learned_motion)
            else:
                # Scheduled sampling
                target_seq = tf.unpack(targets, axis=1)
                
                target_gt_repr_list = []
                with tf.variable_scope(conv_enc_scope, reuse=True):
                    for full_input in input_seq[-1:] + target_seq[:-1]:
                        conv = self._conv_stack(full_input, is_training, memory_device)
                        target_gt_repr_list.append(conv)
                
                inv_sigmoid_decay = tt.training.inverse_sigmoid_decay(1.0, self.global_step,
                                                                      decay_rate=self._scheduled_sampling_decay_rate)
                tf.scalar_summary('scheduled_sampling', inv_sigmoid_decay)
                
                rep_outputs, _ = tt.recurrent.rnn_conv2d_scheduled_sampling(
                    lstm_cell, conv_input_seq[-1],
                    target_gt_repr_list,
                    inv_sigmoid_decay, is_training,
                    initial_state=learned_motion)
                
        # Conv-Decoder   
        with tf.variable_scope("conv-decoder") as varscope:
            deconv_output_seq = []
            for i in xrange(len(rep_outputs)):
                if i > 0:
                    varscope.reuse_variables()

                deconv = self._deconv_stack(rep_outputs[i], input_shape[4],
                                            is_training, memory_device)
                deconv_output_seq.append(deconv)

            # convert back to shape [bs, t, h, w, c]
            packed_result = tf.pack(deconv_output_seq, axis=1)
        
        return packed_result
    
    def _create_lstm_cell(self, input_shape, is_training, memory_device):
        """Creates an LSTMConv2D cell with given input shape and configuration."""
        lstm_cell = tt.recurrent.LSTMConv2DCell(input_shape[0], input_shape[1], input_shape[2],
                                                ksize_input=self._lstm_ksize_input,
                                                ksize_hidden=self._lstm_ksize_hidden,
                                                use_peepholes=self._lstm_use_peepholes,
                                                bn_input_hidden=self._lstm_bn_input_hidden,
                                                bn_hidden_hidden=self._lstm_bn_hidden_hidden,
                                                bn_peepholes=self._lstm_bn_peepholes,
                                                is_training=is_training,
                                                device=memory_device)
        if self._lstm_layers > 1:
            lstm_cell = tt.recurrent.MultiRNNConv2DCell([lstm_cell] * self._lstm_layers)
        return lstm_cell


    def _conv_stack(self, inputs, is_training, memory_device):
        """Creates a 2D convolutional stack."""
        current_inputs = inputs
        for i, (f, k, s) in enumerate(zip(self._filters, self._ksizes, self._strides)):
            conv = tt.network.conv2d("conv{}".format(i + 1),
                                     current_inputs, f, k, s,
                                     weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                     bias_init=self._bias_init,
                                     regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                     activation=tf.nn.relu,
                                     device=memory_device)
            if self._bn_feature_enc:
                conv = tf.contrib.layers.batch_norm(conv, is_training=is_training,
                                                    scope="conv{}_bn".format(i + 1))
            current_inputs = conv

        return current_inputs


    def _deconv_stack(self, inputs, channels, is_training, memory_device):
        """Creates a 2D transposed convolutional stack."""
        rev_filters = self._filters[::-1]
        rev_filters = rev_filters[1:] + [channels]
        rev_ksizes = self._ksizes[::-1]
        rev_strides = self._strides[::-1]
        
        stack_size = len(self._filters)
        current_inputs = inputs
        for i, (f, k, s) in enumerate(zip(rev_filters, rev_ksizes, rev_strides)):
            if self._bn_feature_dec:
                current_inputs = tf.contrib.layers.batch_norm(current_inputs, is_training=is_training,
                                                              scope="deconv{}_bn".format(i + 1))
            deconv = tt.network.conv2d_transpose("deconv{}".format(i + 1),
                                                 current_inputs, f, k, s,
                                                 weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                                 bias_init=self._bias_init,
                                                 regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                                                 device=memory_device)
            # activation
            if i == stack_size - 1:
                current_inputs = self._output_activation(deconv)
            else:
                current_inputs = tf.nn.relu(deconv)

        return current_inputs

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
        bce_last = tt.loss.bce(predictions[:,pshape[1] - 1,:,:,:],
                               targets[:,tshape[1] - 1,:,:,:], name="last_frame")
        tf.add_to_collection(tt.core.LOG_LOSSES, bce_last)
        
        # combine both losses to optimize
        return tf.add(bce_loss, gdl_loss, name="combined_loss")
    
    @tt.utils.attr.override
    def evaluation(self, predictions, targets, device_scope):
        # split as list of element-shape [bs, h, w, c]
        predictions_list = tf.unpack(predictions, axis=1)
        targets_list = tf.unpack(targets, axis=1)
        
        psnr = sharpdiff = ssim = 0.0
        for pred, tgt in zip(predictions_list, targets_list):
            psnr += tt.image.psnr(pred, tgt)
            sharpdiff += tt.image.sharp_diff(pred, tgt)
            ssim += tt.image.ssim(pred, tgt)
            
        # average values
        n = len(targets_list)
        psnr /= n
        sharpdiff /= n
        ssim /= n
            
        # add bce/mse losses here as well to see them in the evaluation
        bce = tt.loss.bce(predictions, targets)
        mse = tt.loss.mse(predictions, targets)

        return {"bce": bce, "mse": mse, "psnr": psnr, "sharpdiff": sharpdiff, "ssim": ssim}