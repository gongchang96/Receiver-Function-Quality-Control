#!/usr/bin/env python
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Conv2DTranspose, add, Concatenate, LSTM
import itertools
import numpy as np
import matplotlib.pyplot as plt

def transformer(drop_rate, width, name, inpC):
    ' Returns a transformer block containing one addetive attention and one feed  forward layer with residual connections '
    x = inpC

    att_layer, weight = SeqSelfAttention(return_attention =True,attention_width = width,name=name)(x)

    #  att_layer = Dropout(drop_rate)(att_layer, training=True)
    att_layer2 = add([x, att_layer])
    norm_layer = LayerNormalization()(att_layer2)

    FF = FeedForward(units=128, dropout_rate=drop_rate)(norm_layer)

    FF_add = add([norm_layer, FF])
    norm_out = LayerNormalization()(FF_add)

    return norm_out, weight

def global_and_local_attention_layer(dropout_rate=0.1, width=None, name=None, input=None, kr = None):
    x = input
    att_G0, _ = transformer(drop_rate=dropout_rate, width = None, name = name + '_att_G0', inpC=x)

    att_G, _ = transformer(drop_rate=dropout_rate, width = None, name = name + '_att_G', inpC=att_G0)

    x_lstm = LSTM(kr, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(att_G)

    att_L, _ = SeqSelfAttention(return_attention=True, attention_width=width, name = name + '_att_L')(x_lstm)

    output = Concatenate(axis=-1)([att_G,att_L])

    return output

class LayerNormalization(tf.keras.layers.Layer):

    """

    Layer normalization layer modified from https://github.com/CyberZHG based on [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

    Parameters
    ----------
    center: bool
        Add an offset parameter if it is True.

    scale: bool
        Add a scale parameter if it is True.

    epsilon: bool
        Epsilon for calculating variance.

    gamma_initializer: str
        Initializer for the gamma weight.

    beta_initializer: str
        Initializer for the beta weight.

    Returns
    -------
    data: 3D tensor
        with shape: (batch_size, …, input_dim)

    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):

        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)


    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs



class FeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward layer. modified from https://github.com/CyberZHG
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y

class SeqSelfAttention(tf.keras.layers.Layer):
    """Layer initialization. modified from https://github.com/CyberZHG
    For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
    :param units: The dimension of the vectors that used to calculate the attention weights.
    :param attention_width: The width of local attention.
    :param attention_type: 'additive' or 'multiplicative'.
    :param return_attention: Whether to return the attention weights for visualization.
    :param history_only: Only use historical pieces of data.
    :param kernel_initializer: The initializer for weight matrices.
    :param bias_initializer: The initializer for biases.
    :param kernel_regularizer: The regularization for weight matrices.
    :param bias_regularizer: The regularization for biases.
    :param kernel_constraint: The constraint for weight matrices.
    :param bias_constraint: The constraint for biases.
    :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                              in additive mode.
    :param use_attention_bias: Whether to use bias while calculating the weights of attention.
    :param attention_activation: The activation used for calculating the weights of attention.
    :param attention_regularizer_weight: The weights of attention regularizer.
    :param kwargs: Parameters for parent class.
    """

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attention_activation = tf.keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = tf.keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': tf.keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'attention_activation': tf.keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}

def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true) + 1

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1.001 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1.001 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def res_block(input_x):
    conv_1 = tf.keras.layers.Conv1D(filters=8,kernel_size=11,padding='same',activation='relu')(input_x)
    conv_2 = tf.keras.layers.Conv1D(filters=8,kernel_size=11,padding='same',activation='relu')(conv_1)
    conv_out = tf.keras.layers.add([conv_1, conv_2])
    drop_out = tf.keras.layers.Dropout(0.1)(conv_out)
    return drop_out

def get_rf_transformer_model(input_shape):
    input_tensor = tf.keras.layers.Input(input_shape)
    res_block_1 = res_block(input_tensor)
    pool_1 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_1)

    res_block_2 = res_block(pool_1)
    pool_2 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_2)

    res_block_3 = res_block(pool_2)
    pool_3 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_3)

    res_block_4 = res_block(pool_3)
    pool_4 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_4)

    bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(pool_4)

    transformer = global_and_local_attention_layer(dropout_rate = 0.1, width = 5, name = 'transformer_block_', input = bilstm_1, kr = 8)

    flatten = tf.keras.layers.Flatten()(transformer)

    dense_1 = tf.keras.layers.Dense(32,activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(8,activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(2,activation='sigmoid')(dense_2)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=dense_3)
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=5.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Recall()])

    return model

def get_rf_LSTM_model(input_shape):
    input_tensor = tf.keras.layers.Input(input_shape)
    res_block_1 = res_block(input_tensor)
    pool_1 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_1)

    res_block_2 = res_block(pool_1)
    pool_2 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_2)

    res_block_3 = res_block(pool_2)
    pool_3 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_3)

    res_block_4 = res_block(pool_3)
    pool_4 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_4)

    bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(pool_4)

    flatten = tf.keras.layers.Flatten()(bilstm_1)

    dense_1 = tf.keras.layers.Dense(32,activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(8,activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(2,activation='sigmoid')(dense_2)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=dense_3)
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=5.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Recall()])

    return model

def get_rf_cnn_model(input_shape):
    input_tensor = tf.keras.layers.Input(input_shape)
    res_block_1 = res_block(input_tensor)
    pool_1 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_1)

    res_block_2 = res_block(pool_1)
    pool_2 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_2)

    res_block_3 = res_block(pool_2)
    pool_3 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_3)

    res_block_4 = res_block(pool_3)
    pool_4 = tf.keras.layers.MaxPooling1D(strides=2)(res_block_4)

    flatten = tf.keras.layers.Flatten()(pool_4)

    dense_1 = tf.keras.layers.Dense(32,activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(8,activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(2,activation='sigmoid')(dense_2)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=dense_3)
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=5.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Recall()])

    return model

def get_FC_model(input_shape):
    input_tensor = tf.keras.layers.Input(input_shape)
    flatten = tf.keras.layers.Flatten()(input_tensor)
    dense_1 = tf.keras.layers.Dense(32,activation='relu')(flatten)
    drop_out_1 = tf.keras.layers.Dropout(0.5)(dense_1)
    dense_2 = tf.keras.layers.Dense(16,activation='relu')(drop_out_1)
    drop_out_2 = tf.keras.layers.Dropout(0.5)(dense_2)
    dense_3 = tf.keras.layers.Dense(4,activation='relu')(drop_out_2)
    drop_out_3 = tf.keras.layers.Dropout(0.5)(dense_3)
    dense_4 = tf.keras.layers.Dense(2,activation='sigmoid')(drop_out_3)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=dense_4)
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=5.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Recall()])

    return model


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(title, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
