import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import random_ops as tfrop

import numpy as np

import scipy.signal

import re
import math
import random
import string

from functools import partial
from termcolor import cprint

from .functions import *
from .layers import *
from .rot_pos_enc import *
#from .upfirdn_2d import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape

def shape(tensor):
    static, dynamic = [tensor.shape.as_list(), tf.shape(tensor)]
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def exists(value):
    return value is not None

def empty(tensor):
    return shape(tensor) == []

def default(value, default):
    return value if exists(value) else default

def chunk_(tensor, chunks, axis=0):
    ts = shape(tensor)
    if axis < 0: axis += len(ts)
    total = ts[axis]
    fit_chunks = total // chunks
    cut = fit_chunks * chunks
    begins = [[0] * len(ts), [0] * axis + [cut] + [0] * (len(ts) - axis - 1)]
    sizes = list(zip(*[[s, s] for s in ts[:axis]] + [[cut, total - cut]] + [[s, s] for s in ts[axis+1:]]))
    main, remainder = tf.slice(tensor, begins[0], sizes[0]), tf.slice(tensor, begins[1], sizes[1])
    chunk = tf.split(main, fit_chunks, axis=axis).append(remainder)
    return chunk

#@contextmanager
#def null_context():
#    yield

def cast_tuple(value):
    return (value,) if not isinstance(value, tuple) else value

def cast_list(value):
    return [value,] if not isinstance(value, list) else value

def shift(token, amount, mask=None):
    if amount == 0: return token
    if exists(mask): token *= tf.broadcast_to(mask, shape(token))
    token_len = shape(token)[1]
    token = tf.pad(token, [[0, 0], [max(amount, 0), max(-amount, 0)]] + [[0, 0]] * (len(shape(token)) - 2))
    return token[:, min(0, -amount):min(0, -amount) + token_len,]

def pre_shift_token(
        token,
        shifts,
        function,
        **kwargs):
    mask = kwargs.get('mask', None)
    segments = len(shifts)
    features_per_shift = shape(token)[-1] // segments
    splitted = tf.split(token, features_per_shift, axis=-1)
    segments_to_shift, rest = splitted[:segments], splitted[segments:]
    segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
    token = tf.conat((*segments_to_shift, *rest), axis=-1)
    return function(token, **kwargs)


# kernel functions
def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, epsilon=1e-4):
    b, i, h, d = shape(data)

    data_normalizer = tf.rsqrt(tf.sqrt(tf.cast(d, data.dtype))) if normalize_data else 1.

    ratio = tf.rsqrt(tf.cast(shape(projection_matrix)[0], data.dtype))

    #data_dash = tf.einsum('...ihd,jd->...ihj', (data_normalizer * data), projection_matrix)
    data_dash = tf.tensordot(data_normalizer * data, projection_matrix, [-1, -1])

    diag_data = data ** 2
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data[Ellipsis, None]

    reduce_axis = -1 if is_query else [-3, -1]
    data_dash = tf.exp(data_dash - diag_data - tf.reduce_max(data_dash, axis=reduce_axis, keepdims=True) + epsilon)
    data_dash *= ratio

    return data_dash


def generalized_kernel(data, *, projection_matrix, kernel_function = tf.nn.relu, normalize_data=True, epsilon=1e-3):
    b, i, h, d = shape(data)

    data_normalizer = tf.rsqrt(tf.sqrt(tf.cast(d, data.dtype))) if normalize_data else 1.

    if not exists(projection_matrix): return kernel_function(data_normalizer * data) + epsilon

    #data_dash = tf.einsum('...ihd,jd->...ihj', (data_normalizer * data), projection_matrix)
    data_dash = tf.tensordot(data_normalizer * data, projection_matrix, [-1, -1])

    data_prime = kernel_function(data_dash) + epsilon

    return data_prime


def orthogonal_matrix_chunk(columns, dtype):
    unstructured_block = tf.random.normal((columns, columns), dtype=dtype)

    q, r = tf.linalg.qr(unstructured_block)

    return tf.transpose(q, [1, 0])


def rotational_products_chunk(columns, dtype):
    rotations = columns * np.ceil(np.log(columns))

    q = np.eye(columns, columns)

    for _ in range(rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(columns, 2)
        index_i, index_j = min(random_indices[0], random_indices[1]), max(random_indices[0], random_indices[1])
        slice_i, slice_j = q[index_i], q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_j + math.cos(random_angle) * slice_j
        q[index_i], q[index_j] = new_slice_i, new_slice_j

    return tf.constant(q, dtype=dtype)


def gaussian_orthogonal_random_matrix(rows, columns, dtype=tf.float64, scaling=0, struct_mode=False):
    full_blocks = int(rows / columns)
    block_list = []

    create_function = rotational_products_chunk if struct_mode else orthogonal_matrix_chunk

    for _ in range(full_blocks):
        q = create_function(columns, dtype)
        block_list.append(q)

    remaining_rows = rows - full_blocks * columns
    if remaining_rows > 0:
        q = create_function(columns, dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = tf.concat(block_list, axis=0)

    if scaling == 0:
        multiplier = tf.norm(tf.random.normal((rows, columns)), axis=1)
    elif scaling == 1:
        multiplier = tf.math.sqrt(float(columns) * tf.ones(rows, dtype=dtype))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    #return tf.einsum('rr,rc->rc', tf.linalg.diag(multiplier), final_matrix)
    return tf.tensordot(tf.linalg.diag(multiplier), final_matrix, [1, 0])


def noncausal_linear_attention(q, k, v, mask=None):
    #k_sum = tf.einsum("...lhd->...hd", k)
    #denominator = 1.0 / tf.einsum("...lhd,...hd->...lh", q, k_sum)
    k_sum = tf.reduce_sum(k, axis=-3, keepdims=True)
    denominator = 1.0 / tf.reduce_sum(q * k_sum, axis=-1, keepdims=True)
    if exists(mask):
        denom_zeros = tf.equal(tf.reduce_max(mask, axis=-1, keepdims=True), 0)
        denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

    #context = tf.einsum("...lhk,...lhv->...hkv", k, v)
    #attention = tf.einsum("...hdo,...lhd,...lh->...lho", context, q, denominator)
    context = tf.reduce_sum(k[Ellipsis, None] * v[Ellipsis, None, :], axis=-4, keepdims=True)
    attention = tf.reduce_sum((q * denominator)[Ellipsis, None] * context, axis=-2)

    return attention


#@tf.custom_gradient     #TODO update gradient for next refer to layers.py, if needed
def causal_linear_attention(q, k, v, mask=None, gate=None, chunks=128, epsilon=1e-6):
    last_k_sum = 0
    last_context_sum = 0
    attention = []

    if exists(gate):
        gate_chunk = chunk_(gate, chunks, axis=-2)
        if exists(mask): mask_chunk = chunk_(mask, chunks, axis=-3)
        for i, q, k, v in enumerate(zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v)))):
            g = gate_chunk[i], r_g = 1.0 - gate_chunk[i]
            g_cat = tf.concat([tf.ones_like(g[Ellipsis, :1, :], g.dtype), g], axis=-2)
            rg_cat = tf.concat([tf.ones_like(r_g[Ellipsis, :1, :], r_g.dtype), r_g], axis=-2)

            g_rev_cump = tf.cumprod(g_cat, axis=-2, exclusive=True, reverse=True)

            k_cat = tf.concat([last_k_sum, k], axis=-3)
            #k_sum = tf.cumsum(tf.einsum("...ch,...ch,...chd->...chd", rg_cat, g_rev_cump, k_cat), axis=-3)
            k_sum = tf.cumsum(rg_cat[Ellipsis, None] * g_rev_cump[Ellipsis, None] * k_cat, axis=-3)
            k_sum = k_sum[Ellipsis, 1:, :, :]
            #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
            denominator = 1.0 / tf.reduce_sum(q * (k_sum + epsilon), axis=-1, keepdims=True)
            if exists(mask):
                denom_zeros = tf.equal(tf.reduce_max(mask_chunk[i], axis=-1, keepdims=True), 0)
                denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

            #context_cat = tf.concat([last_context_sum, tf.einsum("...chk,...chv->...chkv", k, v)], axis=-4)
            #context_sum = tf.cumsum(tf.einsum("...ch,...ch,...chdo->...chdo", rg_cat, g_rev_cump, context_cat), axis=-4)
            #context_sum = context_sum[Ellipsis, 1:, :, :, :]
            context_cat = tf.concat([last_context_sum, k[Ellipsis, None] * v[Ellipsis, None, :]], axis=-4)
            context_sum = tf.cumsum(rg_cat[Ellipsis, None, None] * g_rev_cump[Ellipsis, None, None], context_cat, axis=-4)
            context_sum = context_sum[Ellipsis, 1:, :, :, :]

            #attn = tf.einsum("...chdo,...chd,...ch,->...cho", context_sum, q, denominator)
            attn = tf.reduce_sum(context_sum * q[Ellipsis, None], axis=-2) * denominator

            last_k_sum = k_sum[Ellipsis, -1:, :, :]
            last_context_sum = context_sum[Ellipsis, -1:, :, :, :]
            attention.append(attn)

    else:
        for q, k, v in zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v))):
            k_sum = last_k_sum + tf.cumsum(k, axis=-3)
            #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
            denominator = 1.0 / tf.reduce_sum(q * (k_sum + epsilon), axis=-1, keepdims=True)
            if exists(mask):
                denom_zeros = tf.equal(tf.reduce_max(mask, axis=-1, keepdims=True), 0)
                denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

            #context = tf.einsum("...chk,...chv->...chkv", k, v)
            context = k[Ellipsis, None] * v[Ellipsis, None, :]
            context_sum = last_context_sum + tf.cumsum(context, axis=-4)

            #attn = tf.einsum("...chdo,...chd,...ch->...cho", context_sum, q, denominator)
            attn = tf.reduce_sum(context_sum * q[Ellipsis, None], axis=-2) * denominator

            last_k_sum = k_sum[Ellipsis, -1:, :, :]
            last_context_sum = context_sum[Ellipsis, -1:, :, :, :]
            attention.append(attn)

    return tf.concat(attention, axis=-3)


def fast_attention(
        query,
        key,
        value,
        out_features=None,
        hidden_features=None,
        heads=4,
        mask_query=None,
        mask_key=None,
        mask_value=None,
        causal=False,
        causal_chunks=128,
        gates=True,
        position_embedding=rotary_embedding,
        orthogonal_random_features=True,
        orthogonal_scaling=0,
        kernel_regularization=tf.nn.relu,
        saved_state=None,
        state_distance=None,
        projection_bias=True,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        query_weight_function=None,
        query_bias_function=None,
        key_weight_function=None,
        key_bias_function=None,
        value_weight_function=None,
        value_bias_function=None,
        out_weight_function=None,
        out_bias_function=None,
        trainable=True,
        scope='fast_attn',
        ):
    with tf.variable_scope(scope):
        if not isinstance(heads, int): raise ValueError("The number of heads must be integer,"
                " but given {}".format(type(heads)))
        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        heads = int(heads) if exists(heads) else 1
        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly,"
                " but heads:{} and out_units: {}".format(heads, out_features))
        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly,"
                " but heads:{} and hidden_units: {}".format(heads, hidden_features))
        hidden_depth = hidden_features // heads
        theta_random_features = hidden_depth * int(np.ceil(np.log(hidden_depth)))

        gates = gates and causal

        # material projection
        q, q_m = linear(query, out_features=[heads, hidden_depth], mask=mask_query, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=query_weight_function, bias_function=query_bias_function, trainable=trainable,
                scope='projection_query')
        k, k_m = linear(key, out_features=[heads, hidden_depth], mask=mask_key, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=key_weight_function, bias_function=key_bias_function, trainable=trainable,
                scope='projection_key')
        v, v_m = linear(value, out_features=[heads, hidden_depth], mask=mask_value, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=value_weight_function, bias_function=value_bias_function, trainable=trainable,
                scope='projection_value')

        if q_m is None: q_m = tf.ones_like(q)
        if k_m is None: k_m = tf.ones_like(k)
        if v_m is None: v_m = tf.ones_like(v)

        if gates:
            gate, gate_m = linear(key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul,
                    trainable=trainable, scope='projection_gate')
            gate = tf.math.sigmoid(gate)
            if gate_m is not None: gate *= gate_m
        else:
            gate = None

        # saved_state
        if saved_state is not None:
            ks, vs, k_m_s, v_m_s = saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_mask["value_mask"]
            k_m = tf.concat([tf.ones_like(k_s) if k_m_s is None else k_m_s, k_m], axis=1)
            v_m = tf.concat([tf.ones_like(v_s) if v_m_s is None else v_m_s, v_m], axis=1)

            k, v = tf.concat([k_s, k], axis=1), tf.concat([v_s, v], axis=1)
            sort_k, sort_v = tf.argsort(k_m, axis=1), tf.argsort(v_m, axis=1)
            k, v = tf.gather(k, sort_k, batch_dims=1), tf.gather(v, sort_v, batch_dims=1)
            if isinstance(state_distance, int):
                cut_k, cut_v = tf.maximum(0, shape(k)[1]-state_distance), tf.maximum(0, shape(v)[1] - state_distance)
                k, v, k_m, v_m = k[cut_k:], v[cut_v:], k_m[cut_k:], v_m[cut_v:]

            saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_state["value_mask"] = k, v, k_m, v_m

            if gates:
                gate_s, gate_m_s = saved_state["gate"], saved_state["gate_mask"]
                gate_m = tf.concat([tf.ones_like(gate_s) if gate_m_s is None else gate_m_s, gate_m], axis=1)

                gate = tf.concat([gate_s, gate], axis=1)
                gate = tf.gather(gate, sort_k, batch_dims=1)
                if isinstance(state_distance, int):
                    gate, gate_m = gate[cut_k:], gate_m[cut_k:]

                saved_state["gate"], saved_state["gate_mask"] = gate, gate_m

            input_relation = 'cross'

        # orthogonal_random_features & kernel_regularization
        if not orthogonal_random_features:
            q = tf.softmax(q, axis=-1)
            k = tf.exp(k - tf.reduce_max(k, axis=-3, keepdims=True)) if causal else tf.softmax(k, axis=-3)
        else:
            saved_matrix = tf.get_variable("projection_matrix", shape=[theta_random_features, hidden_depth],
                    dtype=dtype, initializer=tf.initializers.random_normal(0, 1), trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

            if trainable:
                projection_matrix = gaussian_orthogonal_random_matrix(theta_random_features, hidden_depth,
                        dtype=dtype, scaling=orthogonal_scaling, struct_mode=False)

                def update():
                    with tf.control_dependencies([tf.assign(saved_matrix, projection_matrix)]):
                        return tf.identity(projection_matrix)

                condition = tf.abs(tf.reduce_mean(projection_matrix)) > tf.abs(tf.reduce_mean(saved_matrix))
                projection_matrix = tf.cond(condition, update, lambda: tf.identity(projection_matrix))
            else:
                projection_matrix = saved_matrix

            if exists(kernel_regularization):
                q = generalized_kernel(q, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
                k = generalized_kernel(k, projection_matrix=projection_matrix, kernel_function=kernel_regularization)
            else:
                q = softmax_kernel(q, projection_matrix=projection_matrix, is_query=True)
                k = softmax_kernel(k, projection_matrix=projection_matrix, is_query=False)

        # pos emb
        if not gates and exists(position_embedding):
            q = position_embedding(q, q_shape[-2], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
            k = position_embedding(k, k_shape[-2], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

        if causal:
            tensor = causal_linear_attention(q, k, v, mask=q_m, gate=gate, chunks=causal_chunks)
        else:
            tensor = noncausal_linear_attention(q, k, v, mask=q_m)

        t_m = tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, hidden_depth]) if exists(mask_query) else None
        tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=t_m,
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                scope='projection_out')

        tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])

    return tensor, mask_tensor, saved_state



