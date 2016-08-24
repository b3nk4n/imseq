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
    return conv3


def _deconv_stack(representation, reg_lambda, channels):
    deconv1 = tt.network.conv2d_transpose("Deconv1", representation, 64,
                                          (3, 3), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          activation=tf.nn.relu,
                                          device='/cpu:0')
    deconv2 = tt.network.conv2d_transpose("Deconv2", deconv1, 32,
                                          (3, 3), (1, 1),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          activation=tf.nn.relu,
                                          device='/cpu:0')
    deconv3 = tt.network.conv2d_transpose("Deconv3", deconv2, channels,
                                          (3, 3), (2, 2),
                                          weight_init=tf.contrib.layers.xavier_initializer_conv2d(),
                                          bias_init=0.1,
                                          regularizer=tf.contrib.layers.l2_regularizer(reg_lambda),
                                          device='/cpu:0')
    return deconv3


class ConvLSTMConv2DDecoderEncoderModelV2(tt.model.AbstractModel):
    """Model that uses the LSTM decoder/encoder architecture
       with LSTMConv2D instead of normal LSTM cells that processes
       the convoluted representations.
       Additionally, it does NOT use concatenation in reconstruction/frame-decoder,
       but it uses multiple losses:
       - representation_prediction_loss (to ensure the predicted representation is correct)
       - reconstruction loss (main loss)
       - autoencoder-loss (of input and ground-truth output;
                           to further improve reconstruction/frame-decoder)
       
       References: N. Srivastava et al.
                   http://arxiv.org/abs/1502.04681
    """
    def __init__(self, inputs, targets, reg_lambda=5e-4,
                 lstm_layers=1, lstm_ksize_input=(7, 7), lstm_ksize_hidden=(7,7),
                 scope=None):
        self._lstm_layers = lstm_layers
        self._lstm_ksize_input = lstm_ksize_input
        self._lstm_ksize_hidden = lstm_ksize_hidden
        
        self._future_target_representations = None  # [batch, t, 16, 16, 64]
        self._future_pred_representations = None  # [batch, t, 16, 16, 64]
        self._inputs_recon_frames = None  # [batch, t, 64, 64, 1]
        self._targets_recon_frames = None  # [batch, t, 64, 64, 1]
        
        self._scope = scope
        
        super(ConvLSTMConv2DDecoderEncoderModelV2, self).__init__(inputs, targets, reg_lambda)
    
    @tt.utils.attr.lazy_property
    def predictions(self):
        inputs = tf.transpose(self._inputs, [1, 0, 2, 3, 4])
        inputs = tf.split(0, self.input_shape[1], inputs)
        inputs = [tf.squeeze(i, (0,)) for i in inputs]
        
        targets = tf.transpose(self._targets, [1, 0, 2, 3, 4])
        targets = tf.split(0, self.output_shape[1], targets)
        targets = [tf.squeeze(i, (0,)) for i in targets]
        
        with tf.variable_scope("ConvStack") as varscope:
            # present
            representation_inputs = []
            for i in xrange(len(inputs)):
                if i > 0:
                    varscope.reuse_variables()

                representation = _conv_stack(inputs[i], self.reg_lambda)
                representation_inputs.append(representation)
                
            # future
            representation_targets = []
            for i in xrange(len(targets)):
                # note: we are still reusing variables
                representation = _conv_stack(targets[i], self.reg_lambda)
                representation_targets.append(representation)
            self._future_target_representations = tf.pack(representation_targets, axis=1)
                
        representation_shape = representation.get_shape().as_list()

        with tf.variable_scope("encoder-lstm"):
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden)
            _, enc_state = tt.recurrent.rnn_conv2d(lstm_cell, representation_inputs, initial_state=None)  

        # on test/eval: use previously generated frame
        # TODO: use train-code (above) using a longer input-seq (plus a constr-param, that tells the input_length)?
        with tf.variable_scope('decoder-lstm') as varscope:
            lstm_cell = _create_lstm_cell(representation_shape[1], representation_shape[2],
                                          self._lstm_layers, representation_shape[3], 
                                          self._lstm_ksize_input, self._lstm_ksize_hidden)
            
            representation_outputs, _ = tt.recurrent.rnn_conv2d_roundabout(lstm_cell, representation_inputs[-1],
                                                                           sequence_length=self.output_shape[1],
                                                                           initial_state=enc_state)
            self._future_pred_representations = tf.pack(representation_outputs, axis=1)
            
        with tf.variable_scope("DeconvStack") as varscope:
            # future
            prediction_outputs = []
            for i in xrange(len(representation_outputs)):
                if i > 0:
                    varscope.reuse_variables()
                    
                prediction = _deconv_stack(representation_outputs[i],
                                           self.reg_lambda,
                                           self.input_shape[4])
                prediction_outputs.append(prediction)
            predictions_output = tf.pack(prediction_outputs, axis=1)
                
            # future-target
            reconstruction_targets = []
            for i in xrange(len(representation_targets)):
                # hint: we are still resuing variables    
                prediction = _deconv_stack(representation_targets[i],
                                           self.reg_lambda,
                                           self.input_shape[4])
                reconstruction_targets.append(prediction)
            self._targets_recon_frames = tf.pack(reconstruction_targets, axis=1)
                
            # present
            reconstruction_outputs = []
            for i in xrange(len(representation_inputs)):
                # hint: we are still resuing variables    
                prediction = _deconv_stack(representation_inputs[i],
                                           self.reg_lambda,
                                           self.input_shape[4])
                reconstruction_outputs.append(prediction)
            self._inputs_recon_frames = tf.pack(reconstruction_outputs, axis=1)

               
        return predictions_output

    @tt.utils.attr.lazy_property
    def loss(self):
        predictions = self.predictions
        
        # INTERMEDIATE LOSSES
        # repr. prediction loss
        repr_pred_loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(self._future_pred_representations, self._future_target_representations), 2), reduction_indices=[2,3,4]),
            name="repr_pred_loss")
        tf.add_to_collection('intermediate_losses', repr_pred_loss)
        
        inputs_recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(self._inputs_recon_frames, self._inputs), 2), reduction_indices=[2,3,4]),
            name="inputs_recon_loss")
        tf.add_to_collection('intermediate_losses', inputs_recon_loss)
        
        targets_recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(self._targets_recon_frames, self._targets), 2), reduction_indices=[2,3,4]),
            name="targets_recon_loss")
        tf.add_to_collection('intermediate_losses', targets_recon_loss)
        
        
        # MAIN LOSS (PREDICTED FRAME)
        # Calculate the average squared loss per image across the batch
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.sub(predictions, self._targets), 2), reduction_indices=[2,3,4]),
            name="squared_loss")
        return loss
    
    #@override TODO: does lazy loading work with normal overriding w/o repeating code of base class?
    # http://stackoverflow.com/questions/1021464/how-to-call-a-property-of-the-base-class-if-this-property-is-being-overwritten-i
    @tt.utils.attr.lazy_property
    def total_loss(self):
        """Gets the total loss of the model including the regularization losses.
           Implemented as a lazy property.
        Returns
        ----------
        Returns the total oss as a float.
        """
        wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print('wd_losses', len(wd_losses))
        print('wd_losses', wd_losses)
        im_losses = tf.get_collection('intermediate_losses', self._scope)  # TODO: weight to intm.-losses? change weights during train?
        print('im_losses', len(im_losses))
        print('im_losses', im_losses)
        extra_losses = wd_losses + im_losses
        if len(extra_losses) > 0:
            extra_loss = tf.add_n(extra_losses, name="extra_loss")
            total_loss = tf.add(self.loss, extra_loss, name="total_loss")
        else:
            total_loss = tf.identity(self.loss, name="total_loss")

        return total_loss
