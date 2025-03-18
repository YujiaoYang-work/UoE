import torch
import torch.nn as nn
import math
from model.layer.layers_v2 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


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
    batch_ids = torch.arange(batch_size).view(1, batch_size, 1).expand(n_experts, batch_size, k).to(device)
    batch_ids = batch_ids.reshape(-1, k)

    seq_ids, _ = torch.sort(seq_ids, dim=2, descending=False)
    seq_ids_flattened = seq_ids.view(-1, k)

    ids = batch_ids.to(device) * indices_len + seq_ids_flattened.to(device)
    ids = ids.flatten()

    return values, ids, seq_ids

def preprocess(X, route_mat, Mask):
    route_mat = route_mat.transpose(1, 2)
    batch_size, seq_len, d_model = X.shape
    n_expert = route_mat.shape[1]

    input_ = X.view(batch_size * seq_len, d_model)
    mask_ = Mask.view(batch_size * seq_len)

    route_flatten = route_mat.reshape(-1, seq_len)

    max_len = torch.max(torch.sum(route_flatten > 0, dim=1))

    route_mat = route_mat.transpose(0, 1)
    _, ids, seq_ids = topk_indices(route_mat, max_len)

    x = torch.index_select(input_, 0, ids)
    mask = torch.index_select(mask_, 0, ids)

    x = x.view(n_expert, batch_size, max_len, d_model)
    mask = mask.view(n_expert, batch_size, max_len)
    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.cuda(), mask

class SwitchGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.transformer_dim
        self.num_experts = config.attn_n_experts
        self.capacity_factor = config.capacity
        self.epsilon = config.epsilon
        self.w_gate = nn.Linear(self.dim, self.num_experts)
        self.w_gate0 = nn.Linear(self.dim*16, self.num_experts)
        self.topk = config.attn_topk

    def forward(self, X, use_aux_loss=False):
        batch_size, seq_len, d_model = X.shape
        # Compute gate scores

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

def precompute_freqs_cis(dim: int, end: int, constant: float = 10000.0):
    freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # [d/2]
    t = torch.arange(end, device=freqs.device)  # [length]
    freqs = torch.outer(t, freqs).float()  # [length, d/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(device) # [length, d/2]

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-2] * x.shape[-1])
    shape = [1 if i == 0 else d for i, d in enumerate(x.shape)] # (1, length, 1, d/2)

    return freqs_cis.view(*shape) # [1, length, 1, d/2]

def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # [bs, length, head, d/2]
    xq_ = xq_.transpose(1, 2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, length, 1, d/2]
    xq_ = xq_.view(-1, freqs_cis.shape[1], freqs_cis.shape[2], freqs_cis.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # [bs, length, head, d]
    xk_out = torch.view_as_real(xk_.transpose(1, 2) * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        self.head_dim = config.transformer_dim // config.num_heads
        self.n_experts = config.attn_n_experts

    def forward(self, Q, K, V, route_mat, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, route_mat, keep_shape=False)
        dot = scores / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, :, None, :])
        dot = nn.functional.softmax(dot, dim=-1)
        scores = self.drop_attn(dot)
        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, route_mat, keep_shape=False,
                                                    combine_head=False)

        return X

class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        self.head_dim = config.transformer_dim // config.num_heads

        self.n_experts = config.attn_n_experts

    def forward(self, Q, K, V, route_mat, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, route_mat, keep_shape=False)
        dot = scores / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, :, None, :])
        dot = nn.functional.softmax(dot, dim=-1)
        scores = self.drop_attn(dot)
        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, route_mat, keep_shape=False,
                                                    combine_head=False)

        return X

class SparseSelfAttention(nn.Module):
    def __init__(self, config):
        super(SparseSelfAttention, self).__init__()

        self.grad_checkpointing = config.attention_grad_checkpointing
        self.dim = config.transformer_dim
        self.num_head = config.attn_n_experts
        self.head_dim = config.transformer_dim // self.num_head
        self.attn_type = config.attn_type
        self.n_experts = config.attn_n_experts

        self.qk_nope_head_dim = self.dim // self.n_experts // 2
        self.qk_rope_head_dim = self.dim // self.n_experts // 2

        self.freqs_cis_dim = self.qk_rope_head_dim * self.num_head

        self.switch_gate = SwitchGate(config)

        self.W_qkv = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_head * self.head_dim * 3,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config.init_method)

        self.dconv_fc = None
        self.attn = SparseAttention(config)

        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

        self.ff = RowParallelLinearWithMoE(
            self.num_head * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=config.output_layer_init_method,
            skip_bias_add=False)

    def forward(self, X, mask=None):
        batch_size, seq_len, d_model = X.shape
        Y = X

        route_mat, aux_loss = self.switch_gate(X, use_aux_loss=True)
        X, ids, seq_ids, mask = preprocess(X, route_mat, mask)

        QKV, _ = self.W_qkv(X)

        QKV = QKV.view(self.n_experts, batch_size, -1, 3, d_model // self.n_experts)

        Q = QKV[:, :, :, 0, :]
        K = QKV[:, :, :, 1, :]
        V = QKV[:, :, :, 2, :]

        q_nope, q_pe = torch.split(Q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, k_pe = torch.split(K, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.transpose(0, 1).contiguous()
        k_pe = k_pe.transpose(0, 1).contiguous()

        freqs_cis = precompute_freqs_cis(self.freqs_cis_dim, q_pe.shape[-2])
        q_pe, k_pe = apply_rope(q_pe, k_pe, freqs_cis)
        q_pe = q_pe.permute(2, 0, 1, 3)
        k_pe = k_pe.permute(2, 0, 1, 3)
        Q = torch.cat((q_pe, q_nope), dim=-1)
        K = torch.cat((k_pe, k_nope), dim=-1)

        attn_out = self.attn(Q, K, V, route_mat, mask)
        outputs, _ = self.ff(attn_out, keep_shape=False)

        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(Y)
        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        return outs
