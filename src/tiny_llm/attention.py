import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attention_weights = softmax(scores, axis=-1)

    return mx.matmul(attention_weights, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        query = linear(query, self.wq)
        key = linear(key, self.wk)
        value = linear(value, self.wv)

        query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_size))
        key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_size))
        value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_size))

        query = query.swapaxes(-3, -2)
        key = key.swapaxes(-3, -2)
        value = value.swapaxes(-3, -2)

        output = scaled_dot_product_attention_simple(query, key, value, mask=mask)
        output = output.swapaxes(-3, -2)
        output = output.reshape(output.shape[:-2] + (self.hidden_size,))
        output = linear(output, self.wo)
        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    B = query.shape[:-3]
    assert H_q % H == 0, "Number of query heads must be divisible by the number of key/value heads"
    n_repeats = H_q // H

    query = query.reshape(*B, H, n_repeats, L, D)
    key = key.reshape(*B, H, 1, S, D)
    value = value.reshape(*B, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            if L > S:
                raise ValueError("causal attention requires S >= L")
            mask = causal_mask(L, S, scores.dtype)
            scores = scores + mask
        else:
            mask = mx.broadcast_to(mask, (*B, H_q, L, S))
            mask = mask.reshape(*B, 1, H, n_repeats, L, S)
            scores = scores + mask
    output = mx.matmul(softmax(scores, axis=-1), value)
    return output.reshape(expected_shape)


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
