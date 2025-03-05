import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

import sys
import torch.nn.init as init

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    # max_batch_size: int = 32
    max_seq_len: int = 4096
    # dtype: Literal["bf16", "fp8"] = "bf16"
    # vocab_size: int = 512
    dim: int = 64
    inter_dim: int = 128
    moe_inter_dim: int = 16
    # n_layers: int = 2
    n_dense_layers: int = 0
    n_heads: int = 2
    # moe
    n_routed_experts: int = 7
    n_shared_experts: int = 1
    n_activated_experts: int = 3
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32 // 2
    qk_rope_head_dim: int = 32 // 2
    v_head_dim: int = 32
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

    def __init__(self, n_head, d_model, d_head, d_inner, seq_len, n_routed_experts=4, n_shared_experts=1, n_activated_experts=1):
        super().__init__()
        self.dim = d_model
        self.inter_dim = d_inner
        self.moe_inter_dim = d_inner // n_routed_experts
        self.n_heads = n_head
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.q_lora_rank = d_model // 2
        self.kv_lora_rank = d_model // 2
        self.qk_nope_head_dim = d_head // 2
        self.qk_rope_head_dim = d_head // 2
        self.v_head_dim = d_head
        self.max_seq_len = seq_len
        self.original_seq_len = seq_len

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.float32

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        init_method = init.xavier_normal_
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        init_method(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(self.weight) * self.weight


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

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
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
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

class SoftmaxAttention(nn.Module):
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.drop_attn = torch.nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask):

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim=-1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X


class MHA(nn.Module):
    def __init__(self, args: ModelArgs, apply_rope=True):
        super().__init__()

        self.dim = args.dim
        self.num_head = args.n_heads
        self.head_dim = self.dim // self.num_head

        self.apply_rope = apply_rope
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxAttention(self.head_dim)
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, x: torch.Tensor, mems: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        if mems is not None and len(mems) > 0:
            c = torch.cat([mems, x], 1)
        else:
            c = x

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(c))
        V = self.split_heads(self.W_v(c))

        attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        attn_out = self.combine_heads(attn_out)
        out = self.ff(attn_out)

        return out

    def combine_heads(self, X: torch.Tensor):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X: torch.Tensor):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(self, x: torch.Tensor, mems: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()

        if mems is not None and len(mems) > 0:
            c = torch.cat([mems, x], 1)
            freqs_cis_q = freqs_cis[mems.size(1):]
        else:
            c = x
            freqs_cis_q = freqs_cis

        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis_q)
        kv = self.wkv_a(c)

        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            # q = torch.cat([q_nope, q_pe], dim=-1)
            # kv = self.wkv_b(self.kv_norm(kv))
            # kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # self.k_cache[:bsz, start_pos:end_pos] = k
            # self.v_cache[:bsz, start_pos:end_pos] = v
            # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            print("not currently supported")
            pass
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            kv = self.kv_norm(kv)

            k_pe = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) +
                      torch.einsum("bshr,btr->bsht", q_pe, k_pe)) * self.softmax_scale

        scores.masked_fill_(mask[None, :, None, :].squeeze(-1).bool(), -float('inf'))

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            # x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            print("not currently supported")
            pass
        else:
            x = torch.einsum("bsht,btc->bshc", scores, kv)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        x = self.wo(x.flatten(2))

        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))# if self.dim == 7168 else None
        init_method = init.xavier_normal_
        init_method(self.weight)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)

        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, use_v3: bool = True):
        super().__init__()
        self.use_v3 = use_v3
        if self.use_v3:
            self.attn = MLA(args)
        else:
            self.attn = MHA(args)

        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        # self.ffn = nn.Sequential(                   #for ablation experiments
        #     nn.Linear(args.dim, args.inter_dim),
        #     nn.GELU(),
        #     torch.nn.Dropout(p=0.1),
        #     nn.Linear(args.inter_dim, args.dim),
        #     torch.nn.Dropout(p=0.1)
        # )

        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, mems: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:

        x = x + self.attn(self.attn_norm(x), mems, freqs_cis, mask)
        if self.use_v3:
            pass
        x = x.transpose(0, 1).contiguous()
        x = x + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    def __init__(self, layer_id, args: ModelArgs, use_v3: bool = True):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float32
        super().__init__()
        self.use_v3 = use_v3
        self.max_seq_len = args.max_seq_len
        self.layer = Block(layer_id, args, use_v3=use_v3)

        args.max_seq_len *= 2
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
    def forward(self, h: torch.Tensor, dec_attn_mask=None, mems=None) -> torch.Tensor:
        # if self.use_v3:
        h = h.transpose(0, 1).contiguous()
        seqlen = h.size(1)
        start_pos = 0
        if mems is not None and len(mems) > 0:
            # if self.use_v3:
            mems = mems.transpose(0, 1).contiguous()
            seqlen = h.size(1) + mems.size(1)

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        h = self.layer(h, mems, freqs_cis, dec_attn_mask)

        return h

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())

