import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *



def apply_alignment(
        sequence,
        duration,
        tensor_length = None,
        sequence_mask = None,
        sigma = 10.
        ):
    if not exists(tensor_length):
        tensor_length = tf.reduce_sum(duration, axis = -1)
        tensor_length = tf.reduce_max(tensor_length)
        tensor_length = tf.maximum(1., tensor_length)

    sequence_ends = tf.math.cumsum(duration, axis = 1)
    sequence_centers = (sequence_ends - (duration / 2.))

    output_position = tf.cast(tf.range(tensor_length)[None], dtype = sequence.dtype)
    difference = sequence_centers[:, None, :] - output_position[:, :, None]
    logits = - (difference ** 2 / sigma)
    if exists(sequence_mask):
        logits *= sequence_mask[:, None]
        logits -= 1e6 * (1. - sequence_mask)[:, None]

    alignment = tf.reshape(tf.nn.softmax(logits, axis = -1), [shape(duration)[0], -1, shape(duration)[1]])
    alignment = tf.clip_by_value(alignment, 0., 1.)

    return tf.einsum('bln,bnd->bld', alignment, sequence)



def causal_speech_model(
        text,
        duration,
        speaker,
        image,
        features = 512,
        condition_features = 512,
        text_capacity = 10000,
        speaker_capacity = 100000,
        encodings = 1,
        decodings = 1,
        dtype = tf.float32,
        text_mask = None,
        image_mask = None,
        quantization = 0.0,
        trainable = trainable,
        scope = 'arts',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse = reuse):
        core_operation = general_attention
        head = max(features // 128, 1)
        kernel = 3
        weight_groups = 4
        normalization_groups = 2 ** int(np.log2(features) / 2)

        condition = retrieval(
                query = tf.reshape(speaker, (shape(speaker)[0], 1)),
                capacity = speaker_capacity,
                size = condition_features,
                dtype = dtype,
                zero_pas = True,
                scale = True,
                trainable = trainable,
                scope = 'cond'
                )

        with tf.variable_scope('encoder'):
            sequence = retrieval(
                    query = text,
                    capacity = text_capacity,
                    size = features,
                    dtype = dtype,
                    trainable = trainable,
                    scope = 'tknz'
                    )

            encoded = apply_alignment(
                    sequence = sequence,
                    duration = duration,
                    tensor_length = shape(image_mask)[1] if exists(image_mask) else None,
                    sequence_mask = text_mask
                    )

            encoded, _, _, _  = attention_block(
                    tensor = encoded,
                    condition = condition,
                    iteration = encodings,
                    features = features,
                    core_operation = core_operation,
                    head = head,
                    kernel = kernel,
                    weight_groups = weight_groups,
                    normalization_groups = normalization_groups,
                    causality = trainable,
                    mask = reconstruct_mask(features, image_mask),
                    quantization = quantization,
                    trainable = trainable,
                    scope = 'encd'
                    )

        with tf.variable_scope('decoder'):
            decoded, _ = convolution(
                    tensor = image,
                    rank = 1,
                    filters = features,
                    kernel = 1,
                    mask = image_mask,
                    trainable = trainable,
                    scope = 'ItoF'
                    )

            decoded, _, _, _ = attention_block(
                    tensor = decoded,
                    companion = encoded,
                    condition = condition,
                    iteration = decodings,
                    features = features,
                    core_operation = core_operation,
                    head = head,
                    kernel = kernel,
                    weight_groups = weight_groups,
                    normalization_groups = normalization_groups,
                    causality = trainable,
                    mask = reconstruct_mask(features, image_mask),
                    quantization = quantization,
                    trainable = trainable,
                    scope = 'decd'
                    )

            image, _ = convolution(
                    tensor = decoded,
                    rank = 1,
                    filters = shape(image)[-1],
                    kernel = 1,
                    mask = reconstruct(features, mask),
                    trainable = trainable,
                    scope = 'FtoI'
                    )

        return image

