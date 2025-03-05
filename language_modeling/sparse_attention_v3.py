import time

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from layers2D2 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
# from layers2D89 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
import torch.nn.functional as F
from itertools import chain
import sys
from torch import profiler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.init as init
import time
from typing import Optional

def generate_matrix(row, col, target_value, ratio):
    matrix = np.random.rand(row, col) * target_value
    total_elements = row * col
    required_elements = int(ratio * total_elements)
    existing_elements = np.sum(matrix > target_value)
    remaining_elements = required_elements - existing_elements

    while remaining_elements > 0:
        rand_row = np.random.randint(0, row)
        rand_col = np.random.randint(0, col)

        if matrix[rand_row, rand_col] <= target_value:
            matrix[rand_row, rand_col] = np.random.rand() * (1.1 * target_value - target_value) + target_value
            remaining_elements -= 1

    return matrix

def generate_batches(batch_size, row, col, target_value, ratio):
    batches = []
    for _ in range(batch_size):
        batch = generate_matrix(row, col, target_value, ratio)
        batches.append(batch)
    batches = torch.tensor(batches)
    return batches

def split_heads(X, num_head, head_dim):
    X = X.reshape(X.size(0), X.size(1), num_head, head_dim)
    X = X.transpose(1, 2)
    return X

def combine_heads(X, num_head, head_dim):
    X = X.transpose(1, 2)
    X = X.reshape(X.size(0), X.size(1), num_head * head_dim)
    return X

def topk_indices(route_prob, k):
    n_experts, batch_size, indices_len = route_prob.shape
    values, seq_ids = torch.topk(route_prob, k, dim=-1)

    batch_ids = torch.arange(batch_size).view(1, batch_size, 1).expand(n_experts, batch_size, k).to(route_prob.device)
    batch_ids = batch_ids.reshape(-1, k)

    seq_ids, _ = torch.sort(seq_ids, dim=2, descending=False)
    seq_ids_flattened = seq_ids.view(-1, k)

    ids = batch_ids.to(route_prob.device) * indices_len + seq_ids_flattened.to(route_prob.device)
    ids = ids.flatten()

    return values, ids, seq_ids

def preprocess(X, route_mat):
    route_mat = route_mat.transpose(1, 2)
    batch_size, seq_len, d_model = X.shape
    n_expert = route_mat.shape[1]
    input_ = X.view(batch_size * seq_len, d_model)
    route_flatten = route_mat.reshape(-1, seq_len)

    max_len = torch.max(torch.sum(route_flatten > 0, dim=1))
    # max_len = min(max_len, int(seq_len * 0.6))

    route_mat = route_mat.transpose(0, 1)

    _, ids, seq_ids = topk_indices(route_mat, max_len)

    x = torch.index_select(input_, 0, ids)

    x = x.view(n_expert, batch_size, max_len, d_model)
    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.to(x.device)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_kv, sin_kv, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos_q = cos_q[position_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin_q[position_ids].unsqueeze(unsqueeze_dim)
    cos_kv = cos_kv[position_ids].unsqueeze(unsqueeze_dim)
    sin_kv = sin_kv[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_kv) + (rotate_half(k) * sin_kv)

    return q_embed, k_embed

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, n_head, d_model, config):
        super().__init__()
        #patch_size = embed_dim

        self.dim = d_model
        self.num_experts = n_head
        self.capacity_factor = config['capacity']
        self.epsilon = config['epsilon']
        self.w_gate = nn.Linear(self.dim, self.num_experts)
        self.topk = self.num_experts // 2

    def forward(self, X, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        batch_size, seq_len, d_model = X.shape

        X = self.w_gate(X)
        gate_scores = F.softmax(X, dim=-1)

        top_k_scores, top_k_indices = gate_scores.topk(self.topk, dim=-1)

        # # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * batch_size)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            2, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores, None

class SparseAttention(nn.Module):
    def __init__(self, d_head, n_experts, dropout):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=dropout)
        self.head_dim = d_head
        self.n_experts = n_experts

    def forward(self, Q, K, V, route_mat, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, route_mat, keep_shape=False)
        dot = scores / math.sqrt(self.head_dim)

        mask = mask[:dot.size(2), :dot.size(3)]

        dot.masked_fill_(mask[None, None, :, :].squeeze(-1).bool(), -float('inf'))

        dot = nn.functional.softmax(dot, dim=-1)
        scores = self.drop_attn(dot)

        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, route_mat, keep_shape=False,
                                                    combine_head=False)

        return X

class SparseSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, n_head, d_model, d_head, dropout, config, pre_lnorm=False, max_position_embeddings=None, rope_theta=10000.0, **kwargs):
        super(SparseSelfAttention, self).__init__()

        self.grad_checkpointing = False
        self.dim = d_model
        self.num_heads = n_head
        self.head_dim = d_head
        self.num_key_value_heads = n_head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.n_experts = n_head
        if max_position_embeddings is not None:
            self.max_position_embeddings = max_position_embeddings
        else:
            self.max_position_embeddings = d_model // 2
        self.rope_theta = rope_theta
        self.is_causal = True
        self.attention_dropout = dropout  # notice: support inference only.

        if (self.head_dim * self.num_heads) != self.dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        self.switch_gate = SwitchGate(n_head, d_model, config)

        self.W_q = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_heads * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config['init_method'])

        self.W_kv = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_heads * self.head_dim * 2,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config['init_method'])

        self.dconv_fc = None
        self.attn = SparseAttention(d_head, self.n_experts, dropout)

        self.drop_attn = torch.nn.Dropout(p=dropout)

        self.ff = RowParallelLinearWithMoE(
            self.num_heads * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=config['output_layer_init_method'],
            skip_bias_add=False)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x, attn_mask=None, mems=None, position_ids=None):

        batch_size, seq_len, d_model = x.size()

        y = x
        route_mat, aux_loss = self.switch_gate(x, use_aux_loss=True)
        x, ids, seq_ids = preprocess(x, route_mat)

        if mems is not None and len(mems) > 0:
            mems = mems.transpose(0, 1).contiguous()
            route_mat2, aux_loss2 = self.switch_gate(mems, use_aux_loss=True)
            mems, ids2, seq_ids2 = preprocess(mems, route_mat2)
            c = torch.cat([mems, x], 2)
            use_mems = True
        else:
            c = x
            use_mems = False

        query_states, _ = self.W_q(x)
        kv_states, _ = self.W_kv(c)

        query_states = query_states.view(self.n_experts, batch_size, -1, d_model // self.n_experts)
        kv_states = kv_states.view(self.n_experts, batch_size, -1, 2, d_model // self.n_experts)
        key_states = kv_states[:, :, :, 0, :]
        value_states = kv_states[:, :, :, 1, :]

        q_seq_len = query_states.size(-2)
        kv_seq_len = key_states.size(2)
        mem_seq_len = kv_seq_len - q_seq_len

        cos_q, sin_q = self.rotary_emb(query_states, seq_len=q_seq_len)
        cos_kv, sin_kv = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q, cos_kv, sin_kv, position_ids)

        if use_mems:
            m_mask = attn_mask[:q_seq_len, :mem_seq_len]
            h_mask = attn_mask[:q_seq_len, seq_len:seq_len + q_seq_len]
            attn_mask = torch.cat((m_mask, h_mask), dim=1)
        else:
            attn_mask = attn_mask[:q_seq_len, :kv_seq_len]

        attn_out = self.attn(query_states, key_states, value_states, route_mat, attn_mask)

        outputs, _ = self.ff(attn_out, keep_shape=False)

        if (len(outputs.shape) == 3):
            outputs = outputs.view(self.n_experts, batch_size, -1, outputs.shape[2])

        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(y)

        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        if self.pre_lnorm:
            output = y + outs
        else:
            output = self.layer_norm(y + outs)

        return output

