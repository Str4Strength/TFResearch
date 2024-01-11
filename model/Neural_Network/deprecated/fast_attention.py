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
    sizes = list(zip(*[[s, s] for s in ts[:axis]] + [[cut, total - cut]] + [[s, s] for s in ts[axis + 1:]]))
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


# Attention Sub-functions
def process_checker(q_procs, k_procs, v_procs):
    lenproc_q = len(q_procs) if exists(q_procs) else 0
    lenproc_k = len(k_procs) if exists(k_procs) else 0
    lenproc_v = len(v_procs) if exists(v_procs) else 0
    lenproc_kq, lenproc_vk, lenproc_qv, lenproc_qkv = lenproc_q * lenproc_k, lenproc_k * lenproc_v, lenproc_v * lenproc_q, lenproc_q * lenproc_k * lenproc_v

    if lenproc_qkv != 0:
        assert (lenproc_q == lenproc_k) and (lenproc_k == lenproc_v)
    else:
        if lenproc_kq != 0:
            assert lenproc_q == lenproc_k
            v_procs = [tf.identity] * lenproc_k
        elif lenproc_vk != 0:
            assert lenproc_k == lenproc_v
            q_procs = [tf.identity] * lenproc_v
        elif lenproc_qv != 0:
            assert lenproc_v == lenproc_q
            k_procs = [tf.identity] * lenproc_q
        elif lenproc_q != 0:
            k_procs, v_procs = [[tf.identity] * lenproc_q] * 2
        elif lenproc_k != 0:
            v_procs, q_procs = [[tf.identity] * lenproc_k] * 2
        elif lenproc_v != 0:
            q_procs, k_procs = [[tf.identity] * lenproc_v] * 2
        else:
            q_procs, k_procs, v_procs = [[tf.identity]] * 3

    return q_procs, k_procs, v_procs


def split_heads(x, heads, mask=None, **kwargs):
    split_shape = shape(x)[:-1] + [heads, shape(x)[-1] // heads]
    return map(lambda t: tf.reshape(t, split_shape), (x, mask))


def concat_memory(t, t_prev=None, mask=None, mask_prev=None):
    if exists(t_prev):
        t = tf.concat([t_prev, t], axis=1)
        if exists(mask):
            if not exists(mask_prev): mask_prev = tf.ones_like(t_prev, dtype=t.dtype)
            mask = tf.concat([mask_prev, mask], axis=1)
    return t, mask

def sort_linked(t, mask=None):
    if exists(mask):
        sort_kwargs = {'indices': tf.argsort(t, axis=1), 'batch_dims': 1}
        t, mask = map(lambda x: tf.gather(x, **sort_kwargs), (t, mask))
    return t, mask

def link_memory(t, memory_dict, mem_name, mem_mask_name, mask=None, max_distance=None):
    t_mem, t_mem_mask = map(lambda kwd: getattr(memory_dict, kwd, None) (mem_name, mem_mask_name))
    t, t_m = concat_memory(t, t_mem, mask, t_mem_mask)
    t, t_m = sort_linked(t, t_m)
    if isinstance(max_distance, int):
        cut = tf.maximum(0, shape(t)[1] - max_distance)
        t = t[cut:]
        if exists(mask): t_m = t_m[cut:]
    map(lambda kwd, x: setattr(memory_dict, kwd, x), (mem_name, mem_mask_name), (t, t_m))
    return t, t_m


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

    return tf.tensordot(tf.linalg.diag(multiplier), final_matrix, [1, 0])


def get_orthogonal_matrix(depth, dtype, scaling, trainable=True):
    theta_random_features = depth * int(np.ceil(np.log(depth)))
    saved_matrix = tf.get_variable("projection_matrix", shape=[theta_random_features, depth],
            dtype=dtype, initializer=tf.initializers.random_normal(0, 1), trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

    if trainable:
        projection_matrix = gaussian_orthogonal_random_matrix(theta_random_features, depth, dtype=dtype, scaling=scaling, struct_mode=False)
        condition = tf.abs(tf.reduce_mean(projection_matrix)) > tf.abs(tf.reduce_mean(saved_matrix))
        projection_matrix = tf.cond(condition, lambda: update(saved_matrix, projection_matrix), lambda: tf.identity(projection_matrix))
    else:
        projection_matrix = saved_matrix

    return projection_matrix


def get_denom(q, ksum, mask=None, epsilon=0.0):
    denominator = 1.0 / summ(q * (ksum + epsilon), axis=-1, keepdims=True)
    if exists(mask):
        denom_zeros = tf.equal(maxim(mask, axis=-1, keepdims=True), 0)
        denominator = tf.where(denom_zeros, tf.zeros_like(denominator), denominator)

    return denominator


def noncausal_linear_attention(q, k, v, mask=None, epsilon=1e-6):
    #k_sum = tf.einsum("...lhd->...hd", k)
    #denominator = 1.0 / tf.einsum("...lhd,...hd->...lh", q, k_sum)
    k_sum = tf.reduce_sum(k, axis=-3, keepdims=True)
    denominator = get_denom(q, k_sum, mask=mask, epsilon=0.0)

    #context = tf.einsum("...lhk,...lhv->...hkv", k, v)
    #attention = tf.einsum("...hdo,...lhd,...lh->...lho", context, q, denominator)
    context = tf.reduce_sum(k[Ellipsis, None] * v[Ellipsis, None, :], axis=-4, keepdims=True)
    attention = tf.reduce_sum((q * denominator)[Ellipsis, None] * context, axis=-2)

    return attention


#@tf.custom_gradient     #TODO update gradient for next refer to layers.py, if needed
def gated_causal_linear_attention(q, k, v, gate, mask=None, chunks=128, epsilon=1e-6):
    last_k_sum = 0
    last_context_sum = 0
    attention = []

    gate_chunk = chunk_(gate, chunks, axis=-2)
    if exists(mask): mask_chunk = chunk_(mask, chunks, axis=-3)
    for i, q, k, v in enumerate(zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v)))):
        g, r_g = gate_chunk[i], 1.0 - gate_chunk[i]
        g_cat = tf.concat([tf.ones_like(g[Ellipsis, :1, :], g.dtype), g], axis=-2)
        rg_cat = tf.concat([tf.ones_like(r_g[Ellipsis, :1, :], r_g.dtype), r_g], axis=-2)

        g_rev_cump = tf.cumprod(g_cat, axis=-2, exclusive=True, reverse=True)

        k_cat = tf.concat([last_k_sum, k], axis=-3)
        #k_sum = tf.cumsum(tf.einsum("...ch,...ch,...chd->...chd", rg_cat, g_rev_cump, k_cat), axis=-3)
        k_sum = tf.cumsum(rg_cat[Ellipsis, None] * g_rev_cump[Ellipsis, None] * k_cat, axis=-3)
        k_sum = k_sum[Ellipsis, 1:, :, :]
        #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
        denominator = get_denom(q, k_sum, mask=mask, epsilon=epsilon)

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

    return tf.concat(attention, axis=-3)


def causal_linear_attention(q, k, v, mask=None, chunks=128, epsilon=1e-6):
    last_k_sum = 0
    last_context_sum = 0
    attention = []

    for q, k, v in zip(*map(lambda t: chunk_(t, chunks, axis=-3), (q, k, v))):
        k_sum = last_k_sum + tf.cumsum(k, axis=-3)
        #denominator = 1.0 / tf.einsum("...chd,...chd->...ch", q, k_sum + epsilon)
        denominator = get_denom(q, k_sum, mask=mask, epsilon=epsilon)

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
        out_features                    =       None,
        hidden_features                 =       None,
        heads                           =       4,
        mask_query                      =       None,
        mask_key                        =       None,
        mask_value                      =       None,
        causal                          =       False,
        causal_chunks                   =       128,
        gates                           =       True,
        position_embedding              =       rotary_embedding,
        orthogonal_random_features      =       True,
        orthogonal_scaling              =       0,
        kernel_regularization           =       tf.nn.relu,
        saved_state                     =       None,
        state_distance                  =       None,
        projection_bias                 =       True,
        quantization                    =       0.0,
        quantization_blocks             =       8,
        lrmul                           =       1.0,
        query_processes                 =       None,
        key_processes                   =       None,
        value_processes                 =       None,
        component_projection            =       True,
        context_projection              =       True,
        query_weight_function           =       None,
        query_bias_function             =       None,
        key_weight_function             =       None,
        key_bias_function               =       None,
        value_weight_function           =       None,
        value_bias_function             =       None,
        out_weight_function             =       None,
        out_bias_function               =       None,
        trainable                       =       True,
        scope                           =       'fast_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)

        if not isinstance(heads, int): raise ValueError("The number of heads must be integer, but given {}".format(type(heads)))

        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        heads = int(heads) if exists(heads) else 1

        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads
        gates = gates and causal
        if exists(saved_state):
            if exists(getattr(saved_state, 'key', None)): input_relation = 'cross'


        # material projection & construction
        mat_proj_kwargs = {
                'out_features': [heads, hidden_depth],
                'bias': projection_bias,
                'lrmul': lrmul,
                'quantization': quantization,
                'quantization_blocks': quantization_blocks,
                'trainable': trainable,
                }
        material_projector = linear if component_projection else partial(split_heads, heads=heads)

        q, q_m = material_projector(query, mask=mask_query, weight_function=query_weight_function, bias_function=query_bias_function,
                scope='projection_query', **mat_proj_kwargs)

        k, k_m = material_projector(key, mask=mask_key, weight_function=key_weight_function, bias_function=key_bias_function,
                scope='projection_key', **mat_proj_kwargs)
        if exists(saved_state): k, k_m = link_memory(k, saved_state, 'key', 'key_mask', mask=k_m, max_distance=state_distance)

        v, v_m = material_projector(value, mask=mask_value, weight_function=value_weight_function, bias_function=value_bias_function,
                scope='projection_value', **mat_proj_kwargs)
        if exists(saved_state): v, v_m = link_memory(v, saved_state, 'value', 'value_mask', mask=v_m, max_distance=state_distance)

        if gates:
            g, g_m = sigmoid(linear(key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul, trainable=trainable, scope='projection_gate'))
            if exists(saved_state): g, g_m = link_memory(g, saved_state, 'gate', 'gate_mask', mask=g_m, max_distance=state_distance)

        # processing
        processed_qkv = list(map(lambda t, procs: list(map(lambda f: f(t), procs)), (q, k, v), (query_processes, key_processes, value_processes)))

        if orthogonal_random_features:
            projection_matrix = get_orthogonal_matrix(depth=shape(processed_qkv[0][0])[-1], dtype=dtype, scaling=orthogonal_scaling, trainable=trainable)

            if exists(kernel_regularization):
                q_mapping = partial(generalized_kernel, projection_matrix = projection_matrix, kernel_function = kernel_regularization)
                k_mapping = partial(generalized_kernel, projection_matrix = projection_matrix, kernel_function = kernel_regularization)
            else:
                q_mapping = partial(softmax_kernel, projection_matrix = projection_matrix, is_query = True)
                k_mapping = partial(softmax_kernel, projection_matrix = projection_matrix, is_query = False)
        else:
            q_mapping = partial(tf.nn.softmax, axis=-1)
            k_mapping = lambda t: tf.exp(t - tf.reduce_max(t, axis=1, keepdims=True))

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(*processed_qkv))):
            assert shape(q)[-1] == shape(k)[-1]

            q, k = q_mapping(q), k_mapping(k)

            # pos emb
            if not gates and exists(position_embedding):
                #q, k = map(lambda t, s: position_embedding(t, shape(t)[1], shape(t)[-1], trainable=trainable, scope=f'pos_emb_{s}'), (q, k), ('q', 'k'))
                q = position_embedding(q, shape(q)[1], shape(q)[-1], trainable=trainable, scope='pos_emb_q')
                k = position_embedding(k, shape(k)[1], shape(k)[-1], trainable=trainable, scope='pos_emb_k')

            if exists(q_m): q *= tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, shape(q)[-1]])
            if exists(k_m): k *= tf.tile(tf.reduce_max(k_m, axis=-1, keepdims=True), [1, 1, 1, shape(k)[-1]])

            if gates:
                tensor = gated_causal_linear_attention(q, k, v, mask=q_m, gate=gate, chunks=causal_chunks)
            elif causal:
                tensor = causal_linear_attention(q, k, v, mask=q_m, chunks=causal_chunks)
            else:
                tensor = noncausal_linear_attention(q, k, v, mask=q_m)

            mask_tensor = tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, shape(v)[-1]]) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor,
                        bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')
            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensors, masks_tensor, saved_state






def simple_window(tensor, window_size, pad_values=-1):
    b, l, h, d = shape(tensor)
    lp, rp = (window_size - 1) // 2, window_size // 2
    padded = tf.pad(tensor, ((0, 0), (lp, rp), (0, 0), (0, 0)))
    indices = tf.range(l)[:, None] + tf.range(window_size)[None]
    tensor = tf.gather(padded, indices, axis=1, batch_dims=0)
    return tensor


def aligned_window(tensor, window_size, tensor_length, pad_values=-1, mask=None):
    b, l, h, d = shape(tensor)
    ratio = tf.cast(l, dtype=tf.float32) / tf.cast(tensor_length, dtype=tf.float32)
    lp, rp = (window_size - 1) // 2, window_size // 2
    padded = tf.pad(tensor, ((0, 0), (lp, rp), (0, 0), (0, 0)), constant_values=pad_values)
    alignment = tf.cast(tf.math.floor(tf.range(tensor_length, dtype=tf.float32) * ratio), dtype=tf.int32)
    indices = alignment[:, None] + tf.range(window_size)[None]
    tensor = tf.gather(padded, indices, axis=1, batch_dims=0)
    if exists(mask):
        mask = tf.pad(mask, ((0, 0), (lp, rp), (0, 0), (0, 0)), constant_values=0.0)
        mask = tf.gather(mask, indices, axis=1, batch_dims=0)
    return tensor, mask
    

def local_attention(q, k, v, window_size, mask_q=None, mask_k=None, causal=False):
    if not exists(mask_q): mask_q = tf.ones_like(q)
    if not exists(mask_k): mask_k = tf.ones_like(k)

    k, mask_k = aligned_window(k, window_size, shape(q)[1], pad_values=-2.**32, mask=mask_k)     # b q w h d
    v, _ = aligned_window(v, window_size, shape(q)[1], pad_values=-2.**32)     # b q w h o

    mask_attn = maxim(mask_q[Ellipsis, None, :, :] * mask_k, axis=-1, keepdims=True)     # b q w h 1
    if causal:
        mask_attn *= tf.cast([1] * ((window_size - 1) // 2) + [1] + [0] * (window_size // 2), mask_q.dtype)[None, None, :, None, None]
    
    score = summ(q[Ellipsis, None, :, :] * k, axis=-1, keepdims=True)     # b q 1 h d, b q w h d -> b q w h 1
    score -= maxim(score, mask=mask_attn, axis=-3, keepdims=True)
    score = lp_norm(tf.exp(score), p=1, mask=mask_attn, axis=-3)
    score = tf.where(tf.equal(mask_attn, 0.0), tf.zeros_like(score), score)

    attention = summ(score * v, mask=tf.tile(mask_attn, [1, 1, 1, 1, shape(v)[-1]]), axis=-3)     # b q w h 1, b q w h o -> b q h o

    return attention





def proximal_attention(
        query,
        key,
        value,
        window_size                     =       7,
        out_features                    =       None,
        hidden_features                 =       None,
        heads                           =       4,
        mask_query                      =       None,
        mask_key                        =       None,
        mask_value                      =       None,
        causal                          =       False,
        position_embedding              =       rotary_embedding,
        saved_state                     =       None,
        state_distance                  =       None,
        projection_bias                 =       True,
        quantization                    =       0.0,
        quantization_blocks             =       8,
        lrmul                           =       1.0,
        query_processes                 =       None,
        key_processes                   =       None,
        value_processes                 =       None,
        component_projection            =       True,
        context_projection              =       True,
        query_weight_function           =       None,
        query_bias_function             =       None,
        key_weight_function             =       None,
        key_bias_function               =       None,
        value_weight_function           =       None,
        value_bias_function             =       None,
        out_weight_function             =       None,
        out_bias_function               =       None,
        trainable                       =       True,
        scope                           =       'prox_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)
        
        if not isinstance(heads, int): raise ValueError("The number of heads must be integer, but given {}".format(type(heads)))

        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        heads = int(heads) if exists(heads) else 1

        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads

        if exists(saved_state):
            if exists(getattr(saved_state, 'key', None)): input_relation = 'cross'

        # material projection & construction
        mat_proj_kwargs = {
                'out_features': [heads, hidden_depth],
                'bias': projection_bias,
                'lrmul': lrmul,
                'quantization': quantization,
                'quantization_blocks': quantization_blocks,
                'trainable': trainable,
                }
        material_projector = linear if component_projection else partial(split_heads, heads=heads)

        q, q_m = material_projector(query, mask=mask_query, weight_function=query_weight_function, bias_function=query_bias_function,
                scope='projection_query', **mat_proj_kwargs)

        k, k_m = material_projector(key, mask=mask_key, weight_function=key_weight_function, bias_function=key_bias_function,
                scope='projection_key', **mat_proj_kwargs)
        if exists(saved_state): k, k_m = link_memory(k, saved_state, 'key', 'key_mask', mask=k_m, max_distance=state_distance)

        v, v_m = material_projector(value, mask=mask_value, weight_function=value_weight_function, bias_function=value_bias_function,
                scope='projection_value', **mat_proj_kwargs)
        if exists(saved_state): v, v_m = link_memory(v, saved_state, 'value', 'value_mask', mask=v_m, max_distance=state_distance)

        # processing
        processed_qkv = list(map(lambda t, procs: list(map(lambda f: f(t), procs)), (q, k, v), (query_processes, key_processes, value_processes)))

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(*processed_qkv))):
            assert shape(q)[-1] == shape(k)[-1]

            # pos emb
            if exists(position_embedding):
                q, k = map(lambda t, s: position_embedding(t, shape(t)[1], hidden_depth, trainable=trainable, scope=f'pos_emb_{s}'), (q, k), ('q', 'k'))

            tensor = local_attention(q, k, v, window_size=window_size, mask_q=q_m, mask_k=k_m, causal=causal)
            mask_tensor = tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, shape(v)[-1]]) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor,
                        bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')
            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensors, masks_tensor, saved_state





def hybrid_fast_attention(
        query,
        key,
        value,
        window_size                     =       7,
        out_features                    =       None,
        hidden_features                 =       None,
        full_heads                      =       4,
        prox_heads                      =       4,
        mask_query                      =       None,
        mask_key                        =       None,
        mask_value                      =       None,
        causal                          =       False,
        gates                           =       True,
        position_embedding              =       rotary_embedding,
        orthogonal_random_features      =       True,
        orthogonal_scaling              =       0,
        kernel_regularization           =       tf.nn.relu,
        saved_state                     =       None,
        state_distance                  =       None,
        projection_bias                 =       True,
        quantization                    =       0.0,
        quantization_blocks             =       8,
        lrmul                           =       1.0,
        query_processes                 =       None,
        key_processes                   =       None,
        value_processes                 =       None,
        component_projection            =       True,
        context_projection              =       True,
        query_weight_function           =       None,
        query_bias_function             =       None,
        key_weight_function             =       None,
        key_bias_function               =       None,
        value_weight_function           =       None,
        value_bias_function             =       None,
        out_weight_function             =       None,
        out_bias_function               =       None,
        trainable                       =       True,
        scope                           =       'prox_attn',
        ):
    with tf.variable_scope(scope):
        query_processes, key_processes, value_processes = process_checker(query_processes, key_processes, value_processes)

        if not isinstance(full_heads, int): raise ValueError("The number of full_heads must be integer, but given {}".format(type(full_heads)))
        if not isinstance(prox_heads, int): raise ValueError("The number of prox_heads must be integer, but given {}".format(type(prox_heads)))

        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        heads = int(full_heads + prox_heads)

        out_features = int(out_features) if exists(out_features) else q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly, but heads:{} and out_units: {}".format(heads, out_features))

        hidden_features = int(hidden_features) if exists(hidden_features) else out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly, but heads:{} and hidden_units: {}".format(heads, hidden_features))

        hidden_depth = hidden_features // heads
        gates = gates and causal
        if exists(saved_state):
            if exists(getattr(saved_state, 'key', None)): input_relation = 'cross'


        # material projection & construction
        mat_proj_kwargs = {
                'out_features': [heads, hidden_depth],
                'bias': projection_bias,
                'lrmul': lrmul,
                'quantization': quantization,
                'quantization_blocks': quantization_blocks,
                'trainable': trainable,
                }
        material_projector = linear if component_projection else partial(split_heads, heads=heads)

        q, q_m = material_projector(query, mask=mask_query, weight_function=query_weight_function, bias_function=query_bias_function,
                scope='projection_query', **mat_proj_kwargs)

        k, k_m = material_projector(key, mask=mask_key, weight_function=key_weight_function, bias_function=key_bias_function,
                scope='projection_key', **mat_proj_kwargs)
        if exists(saved_state): k, k_m = link_memory(k, saved_state, 'key', 'key_mask', mask=k_m, max_distance=state_distance)

        v, v_m = material_projector(value, mask=mask_value, weight_function=value_weight_function, bias_function=value_bias_function,
                scope='projection_value', **mat_proj_kwargs)
        if exists(saved_state): v, v_m = link_memory(v, saved_state, 'value', 'value_mask', mask=v_m, max_distance=state_distance)

        if gates:
            g, g_m = sigmoid(linear(key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul, trainable=trainable, scope='projection_gate'))
            if exists(saved_state): g, g_m = link_memory(g, saved_state, 'gate', 'gate_mask', mask=g_m, max_distance=state_distance)

        # processing
        processed_qkv = list(map(lambda t, procs: list(map(lambda f: f(t), procs)), (q, k, v), (query_processes, key_processes, value_processes)))

        if orthogonal_random_features:
            projection_matrix = get_orthogonal_matrix(depth=shape(processed_qkv[0][0])[-1], dtype=dtype, scaling=orthogonal_scaling, trainable=trainable)
            if exists(kernel_regularization):
                q_mapping, k_mapping = [partial(generalized_kernel, projection_matrix = projection_matrix, kernel_function = kernel_regularization)] * 2
            else:
                q_mapping = partial(softmax_kernel, projection_matrix = projection_matrix, is_query = True)
                k_mapping = partial(softmax_kernel, projection_matrix = projection_matrix, is_query = False)
        else:
            q_mapping = partial(tf.nn.softmax, axis=-1)
            k_mapping = lambda t: tf.exp(t - tf.reduce_max(t, axis=1, keepdims=True))

        tensors, masks_tensor = [], []
        for n, (q, k, v) in enumerate(list(zip(*processed_qkv))):
            assert shape(q)[-1] == shape(k)[-1]

            # full
            q_full, k_full, v_full = q[Ellipsis, :full_heads, :], k[Ellipsis, :full_heads, :], v[Ellipsis, :full_heads, :]
            q_full, k_full = q_mapping(q_full), k_mapping(k_full)

            # pos emb
            if not gates and exists(position_embedding):
                q_full, k_full = map(lambda t, s: position_embedding(t, shape(t)[1], shape(t)[-1], trainable=trainable, scope=f'pos_emb_{s}'),
                (q_full, k_full), ('q', 'k'))

            if exists(q_m): q_full *= tf.tile(tf.reduce_max(q_m[Ellipsis, :full_heads, :], axis=-1, keepdims=True), [1, 1, 1, shape(q_full)[-1]])
            if exists(k_m): k_full *= tf.tile(tf.reduce_max(k_m[Ellipsis, :full_heads, :], axis=-1, keepdims=True), [1, 1, 1, shape(k_full)[-1]])

            if gates:
                full = gated_causal_linear_attention(q_full, k_full, v_full, mask=q_m[Ellipsis, :full_heads, :], gate=gate, chunks=causal_chunks)
            elif causal:
                full = causal_linear_attention(q_full, k_full, v_full, mask=q_m[Ellipsis, :full_heads, :], chunks=causal_chunks)
            else:
                full = noncausal_linear_attention(q_full, k_full, v_full, mask=q_m[Ellipsis, :full_heads, :])

            #mask_full = tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, shape(v_full)[-1]]) if exists(mask_query) else None

            # prox
            prox = local_attention(q[Ellipsis, full_heads:, :], k[Ellipsis, full_heads:, :], v[Ellipsis, full_heads:, :], window_size=window_size,
                mask_q=q_m[Ellipsis, full_heads:, :], mask_k=k_m[Ellipsis, full_heads:, :], causal=causal)

            tensor = tf.concat([full, prox], axis=-2)
            mask_tensor = tf.tile(tf.reduce_max(q_m, axis=-1, keepdims=True), [1, 1, 1, shape(v)[-1]]) if exists(mask_query) else None

            if context_projection:
                tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features, mask=mask_tensor,
                        bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                        weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                        scope='projection_out')
            else:
                tensor, mask_tensor = map(lambda t: tf.reshape(t, shape(t)[:-2] + [shape(t)[-2] * shape(t)[-1]]) if exists(t) else None, (tensor, mask_tensor))

            tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
            tensors.append(tensor)
            masks_tensor.append(mask_tensor)

        if len(tensors) == 1: tensors, masks_tensor = *tensors, *masks_tensor

    return tensors, masks_tensor, saved_state




