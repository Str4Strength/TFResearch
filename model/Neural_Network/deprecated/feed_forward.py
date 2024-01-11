import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .functions import *
from .linear import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def feed_forward(
        tensor,
        features,
        hidden_features=None,
        mask=None,
        activation=gelu,
        dropout_rate=0.,
        lrmul=1.0,
        quantization=0.0,
        weight_function=None,
        bias_function=None,
        trainable=True,
        scope='feed_fwd'
        ):
    with tf.variable_scope(scope):
        if hidden_features is None: hidden_features = 4 * features

        tensor, mask = linear(tensor, out_features=hidden_features, mask=mask, lrmul=lrmul, quantization=quantization,
                quantization_blocks=shape(tensor)[-1], weight_function=weight_function, bias_function=bias_function,
                trainable=trainable, scope='fc0')

        tensor = activation(tensor)
        if mask is not None: tensor *= mask

        if dropout_rate > 0.: tensor = tf.nn.dropout(tensor, rate=dropout_rate)

        tensor, mask = linear(tensor, out_features=features, mask=mask, lrmul=lrmul, quantization=quantization,
                quantization_blocks=hidden_features, weight_function=weight_function, bias_function=bias_function,
                trainable=trainable, scope='fc1')

        if dropout_rate > 0.: tensor = tf.nn.dropout(tensor, rate=dropout_rate)

        return tensor, mask



