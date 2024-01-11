import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *
from .rot_pos_enc import *
from .linear import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape



def link_memory(tensor, mask = None, tensor_saved = None, mask_saved = None, distance = None):
    if exists(tensor_saved):
        tensor = tf.concat([tensor_saved, tensor], axis = 1)
        if exists(mask) and not exists(mask_saved): mask = tf.concat([tf.ones_like(tensor_saved, dtype = mask.dtype), mask], axis = 1)
        if not exists(mask) and exists(mask_saved): mask = tf.concat([mask_saved, tf.ones_like(tensor, dtype = mask_saved.dtype)], axis = 1)
        if exists(mask):
            sorting = tf.argsort(mask, axis = 1)
            tensor, mask = tf.gather(tensor, sorting, batch_dims = 1), tf.gather(mask, sorting, batch_dims = 1)
        if isinstance(distance, int):
            cutting = tf.maximum(0, shape(tensor)[1] - distance)
            tensor = tensor[cutting:]
            if exists(mask): mask = mask[cutting:]
    return tensor, mask



def general_attention(q, k, v, sharpness, mask = None, attn_map_bias = None):
    q_dot_k = tf.einsum('...qhd,...khd->...qkh', q, k)
    q_dot_k *= sharpness
    if exists(attn_map_bias): q_dot_k +=  attn_map_bias
    if exists(mask): q_dot_k = tf.where(tf.equal(mask, 0.), -1e9 * tf.ones_like(q_dot_k), q_dot_k)
    attn_weight = tf.nn.softmax(q_dot_k, axis = -2)
    tensor = tf.einsum('...qkh,...khd->...qhd', attn_weight, v)
    return tensor, attn_weight



def l2_attention(q, k, v, sharpness, mask = None, attn_map_bias = None):
    squared_dist = tf.einsum('...qhd,...qhd->...qh', q, q)[Ellipsis, None, :]
    squared_dist +=  tf.einsum('...khd,...khd->...kh', k, k)[Ellipsis, None, :, :]
    squared_dist -=  2 * tf.einsum('...qhd,...khd->...qkh', q, k)
    squared_dist *= sharpness
    if exists(attn_map_bias): squared_dist +=  attn_map_bias
    if exists(mask): squared_dist = tf.where(tf.equal(mask, 0.), -1e9 * tf.ones_like(squared_dist), squared_dist)
    attn_weight = tf.nn.softmax(squared_dist, axis = -2)
    tensor = tf.einsum('...qkh,...khd->...qhd', attn_weight, v)
    return tensor, attn_weight



def cosine_attention(q, k, v, sharpness, mask = None, attn_map_bias = None):
    q, k = map(lambda t: tf.nn.l2_normalize(t, axis = -1, epsilon = 1e-12), (q, k))
    cosine_sim = tf.math.cos(tf.einsum('...qhd,...khd->...qkh', q, k)) * sharpness
    if exists(attn_map_bias): cosine_sim += attn_map_bias
    if exists(mask): cosine_sim *= mask
    tensor = tf.einsum('...qkh,...khd->...qhd', cosine_sim, v)
    return tensor, cosine_sim



def window_sliding(tensor, window, target_size, causality = False, pad_values = -1):
    tensor_size, n_batch_dims = shape(tensor)[-3], len(shape(tensor)) - 3

    ratio = tf.cast(tensor_size, dtype = tensor.dtype) / tf.cast(target_size, dtype = tensor.dtype)
    window = tf.maximum(window, tf.cast(tf.math.ceil(ratio), dtype = tf.int32))

    alignment = tf.math.floor(tf.range(target_size, dtype = tensor.dtype) * ratio)
    indices = tf.cast(alignment, dtype = tf.int32)[:, None] + tf.range(window)[None]

    if causality:
        left_pad, right_pad = window - 1, 0
    else:
        left_pad, right_pad = (window - 1) // 2, window // 2
    paddings = ( *([(0, 0)] * n_batch_dims), (left_pad, right_pad), (0, 0), (0, 0) )
    padded = tf.pad(tensor, paddings = paddings, constant_values = pad_values)
    tensor = tf.gather(padded, indices, axis = -3, batch_dims = n_batch_dims - 1)

    return tensor



def window_attention(q, k, v, sharpness, mask = None, attn_map_bias = None):
    q_dot_k = tf.einsum('...thd,...tkhd->...tkh', q, k)
    q_dot_k *= sharpness
    if exists(attn_map_bias): q_dot_k += attn_map_bias
    if exists(mask): q_dot_k = tf.where(tf.equal(mask, 0.), -1e9 * tf.ones_like(q_dot_k), q_dot_k)
    attn_weight = tf.nn.softmax(q_dot_k, axis = -2)
    tensor = tf.einsum('...tkh,...tkhd->...thd', attn_weight, v)
    return tensor, attn_weight



def attention(
        query,
        key,
        value,
        out_features = None,
        hidden_features = None,
        core_operation = general_attention,
        head = 4,
        window = 9,
        mask_query = None,
        mask_key = None,
        mask_value = None,
        causality = False,
        saved_state = None,
        sharpness = False,
        sharpness_maximum = 1e4,
        sharpness_minimum = 1e-4,
        position_embedding = rotary_embedding,
        attention_bias = None,
        projection_bias = True,
        dropout = 0.0,
        quantization = 0.0,
        quantization_blocks = 8,
        lrmul = 1.0,
        component_projection = True,
        context_projection = True,
        query_weight_function = None,
        query_bias_function = None,
        key_weight_function = None,
        key_bias_function = None,
        value_weight_function = None,
        value_bias_function = None,
        out_weight_function = None,
        out_bias_function = None,
        trainable = True,
        scope = 'attention'
        ):
    with tf.variable_scope(scope):
        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        if out_features is None: out_features = q_shape[-1]
        if hidden_features is None: hidden_features = out_features
        hidden_depth = hidden_features // head


        # linear layers for query, key, value : B, S, C --> B, S, H, D
        if component_projection:
            q, q_m = linear(
                    query, out_features = [head, hidden_depth], mask = mask_query, bias = projection_bias,
                    lrmul = lrmul, quantization = quantization, quantization_blocks = quantization_blocks,
                    weight_function = query_weight_function, bias_function = query_bias_function,
                    trainable = trainable, scope = 'projection_query')

            k, k_m = linear(
                    key, out_features = [head, hidden_depth], mask = mask_key, bias = projection_bias,
                    lrmul = lrmul, quantization = quantization, quantization_blocks = quantization_blocks,
                    weight_function = key_weight_function, bias_function = key_bias_function,
                    trainable = trainable, scope = 'projection_key')

            v, v_m = linear(
                    value, out_features = [head, hidden_depth], mask = mask_value, bias = projection_bias,
                    lrmul = lrmul, quantization = quantization, quantization_blocks = quantization_blocks,
                    weight_function = value_weight_function, bias_function = value_bias_function,
                    trainable = trainable, scope = 'projection_value')
        else:
            q = tf.reshape(query , q_shape[:-1] + [head, hidden_features])
            q_m = tf.reshape(mask_query, q_shape[:-1] + [head, hidden_features]) if exists(mask_query) else None

            k = tf.reshape(key, k_shape[:-1] + [head, hidden_features])
            k_m = tf.reshape(mask_key, k_shape[:-1] + [head, hidden_features]) if exists(mask_key) else None

            v = tf.reshape(value, v_shape[:-1] + [head, hidden_features])
            v_m = tf.reshape(mask_value, v_shape[:-1] + [head, hidden_features]) if exists(mask_value) else None

        if exists(saved_state):
            if exists(getattr(saved_state, 'key', None)): input_relation = 'cross'
            distance = getattr(saved_state, 'distance', None)

            k_saved, k_m_saved = getattr(saved_state, 'key', None), getattr(saved_state, 'key_mask', None)
            k, k_m = link_memory(k, k_m, tensor_saved = k_saved, mask_saved = k_m_saved, distance = distance)
            setattr(saved_state, 'key', k)
            setattr(saved_state, 'key_mask', k_m)


            v_saved, v_m_saved = getattr(saved_state, 'value', None), getattr(saved_state, 'value_mask', None)
            v, v_m = link_memory(v, v_m, tensor_saved = v_saved, mask_saved = v_m_saved, distance = distance)
            setattr(saved_state, 'value', v)
            setattr(saved_state, 'value_mask', v_m)

        # position embedding
        if exists(position_embedding):
            q = position_embedding(q, q_shape[-2], hidden_depth, trainable = trainable, scope = 'pos_emb_q')
            k = position_embedding(k, k_shape[-2], hidden_depth, trainable = trainable, scope = 'pos_emb_k')

        sharpness = weight_(
                shape = [],
                dtype = q.dtype,
                init = tf.ones_initializer,
                lrmul = lrmul,
                trainable = trainable,
                scope = 'sharpness',
                ) if sharpness else 1.
        sharpness = sharpness * tf.rsqrt(tf.cast(shape(q)[-1], q.dtype))
        if exists(sharpness_maximum): sharpness = tf.minimum(sharpness, sharpness_maximum)
        if exists(sharpness_minimum): sharpness = tf.maximum(sharpness, sharpness_minimum)


        mask_attention = tf.reduce_max(q_m, axis = -1) if exists(mask_query) else tf.ones(shape(q)[:-1], dtype = dtype)

        # masked softmax
        if core_operation == window_attention:
            k = window_sliding(k, window = window, target_size = q_shape[-2], causality = causality, pad_values = -1e9)
            v = window_sliding(v, window = window, target_size = q_shape[-2], causality = causality, pad_values = -1e9)

            relative_position_bias = bias_(
                    shape = [2, window, head, hidden_depth],
                    dtype = q.dtype,
                    lrmul = lrmul,
                    trainable = trainable,
                    scope = 'relative_position_bias'
                    )
            rpb_pad = shape(k)[-3] - window
            front = tf.tile(relative_position_bias[:, :1], (1, tf.maximum((rpb_pad - 1) // 2, 0), 1, 1))
            back = tf.tile(relative_position_bias[:, -1:], (1, rpb_pad // 2, 1, 1))
            relative_position_bias = tf.concat([front, relative_position_bias, back], axis = 1)

            k += relative_position_bias[0][None, None]
            v += relative_position_bias[1][None, None]

            mask_attention = tf.tile(mask_attention[Ellipsis, None, :], (1, 1, window, 1))
            #if causality:
            #    mask_causality = tf.cast([1.] * (((window - 1) // 2) + 1) + [0.] * (window // 2), dtype = dtype)
            #    mask_attention *= mask_causality[None, None, :, None]

            tensor, attn_weight = window_attention(
                    q = q,
                    k = k,
                    v = v,
                    sharpness = sharpness,
                    mask = mask_attention,
                    attn_map_bias = attention_bias
                    )

        else:
            mask_attention = tf.einsum('bqh,bkh->bqkh', mask_attention,
                    tf.reduce_max(k_m, axis = -1) if exists(mask_key) else tf.ones(shape(k)[:-1], dtype = dtype))
            if causality:
                mask_causality = tf.cast(tf.range(q_shape[-2])[:, None] >= tf.range(k_shape[-2])[None], dtype = dtype)
                mask_attention *= mask_causality[None, Ellipsis, None]

            tensor, attn_weight = core_operation(
                    q = q,
                    k = k,
                    v = v,
                    sharpness = sharpness,
                    mask = mask_attention,
                    attn_map_bias = attention_bias
                    )

        if context_projection:
            tensor, mask_tensor = linear(
                    tensor, in_features = shape(tensor)[-2:], out_features = out_features,
                    mask = q_m, bias = projection_bias, lrmul = lrmul, quantization = quantization,
                    quantization_blocks = hidden_features * quantization_blocks // q_shape[-1],
                    weight_function = out_weight_function, bias_function = out_bias_function,
                    trainable = trainable, scope = 'projection_out')
        else:
            tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, q_m))

        return tensor, mask_tensor, attn_weight



