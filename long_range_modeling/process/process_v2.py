import torch
from torch import nn
import math
from typing import Literal
import torch.nn.functional as F
from model.layer.linear import linear
import torch.nn.init as init

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoPEArgs:
    qk_rope_head_dim: int = 32
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.



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

def preprocess(X, route_mat, Mask=None, freqs_cis=None):
    route_mat = route_mat.transpose(1, 2)
    batch_size, seq_len, d_model = X.shape
    n_expert = route_mat.shape[1]

    input_ = X.view(batch_size * seq_len, d_model)


    route_flatten = route_mat.reshape(-1, seq_len)

    max_len = torch.max(torch.sum(route_flatten > 0, dim=1))

    # max_len = seq_len // 2
    # max_len = int(seq_len * 0.6)

    route_mat = route_mat.transpose(0, 1)
    _, ids, seq_ids = topk_indices(route_mat, max_len)

    x = torch.index_select(input_, 0, ids)
    if Mask is not None:
        mask_ = Mask.view(batch_size * seq_len)
        mask = torch.index_select(mask_, 0, ids)
        mask = mask.view(n_expert, batch_size, max_len)
    else:
        mask = None
    x = x.view(n_expert, batch_size, max_len, d_model)

    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.cuda(), mask

class SwitchGate(nn.Module):
    def __init__(self, config, is_attn=True):
        super().__init__()
        #注意：常规设置：patch_size = embed_dim
        self.dim = config.transformer_dim
        self.capacity_factor = config.capacity
        self.epsilon = config.epsilon
        self.n_groups = 1
        self.topk_groups = 1

        if is_attn:
            self.num_experts = config.attn_n_experts
            self.topk = config.attn_topk
        else:
            self.num_experts = config.n_experts
            self.topk = config.topk

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

        init_method = init.xavier_normal_
        init_method(self.weight)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, X):
        gate_scores = F.softmax(linear(X, self.weight, self.bias), dim=-1)
        gate_scores_raw = gate_scores
        if self.n_groups > 1:
            gate_scores = gate_scores.view(X.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = gate_scores.amax(dim=-1)
            else:
                group_scores = gate_scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(gate_scores[..., 0]).scatter_(1, indices, True)
            gate_scores = (gate_scores * mask.unsqueeze(-1)).flatten(1)

        top_k_scores, top_k_indices = gate_scores.topk(self.topk, dim=-1)

        mask = torch.zeros_like(gate_scores).scatter_(
            2, top_k_indices, 1
        )
        masked_gate_scores = gate_scores_raw * mask
        denominators = (masked_gate_scores.sum(-1, keepdim=True) + self.epsilon)
        gate_scores = (masked_gate_scores / denominators) * self.capacity_factor

        return gate_scores

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(self.weight) * self.weight


def precompute_freqs_cis(dim, seqlen, original_seq_len=None, beta_fast=32, beta_slow=1, base=10000.0, factor=40) -> torch.Tensor:
    if original_seq_len is None:
        original_seq_len = seqlen
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

