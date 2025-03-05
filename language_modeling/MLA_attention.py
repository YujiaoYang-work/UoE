import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MQA_Attention(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.num_kv_heads = (
            model_args.num_heads if model_args.num_kv_heads == 0 else model_args.num_kv_heads
        )
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_heads)
        self.query = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.proj = nn.Linear(d_model, d_model, model_args.bias)
        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)
        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = k.view(
            batch, seq_len, -1, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, -1, self.head_dim)
        v = v.view(batch, seq_len, -1, self.head_dim)
        q, k = apply_rope(q, k, freqs_cis)
        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)
        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,  # order impotent
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Attention(nn.Module):
    def __init__(self, model_args):

        super().__init__()
        d_model = model_args['d_model']
        self.num_heads = model_args['num_heads']
        self.head_dim = model_args['d_model'] // model_args['num_heads']
        self.attn_dropout = nn.Dropout(model_args['dropout'])
        self.res_dropout = nn.Dropout(model_args['dropout'])
        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.q_lora_rank = model_args['q_lora_rank']
        self.qk_rope_head_dim = model_args['qk_rope_head_dim']
        self.kv_lora_rank = model_args['kv_lora_rank']
        self.v_head_dim = model_args['v_head_dim']
        self.qk_nope_head_dim = model_args['qk_nope_head_dim']
        self.q_head_dim = model_args['qk_nope_head_dim'] + model_args['qk_rope_head_dim']
        self.q_a_proj = nn.Linear(d_model, model_args['q_lora_rank'], bias=False)
        self.q_a_layernorm = RMSNorm(model_args['q_lora_rank'])
        self.q_b_proj = nn.Linear(model_args['q_lora_rank'], self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d_model ,model_args['kv_lora_rank'] + model_args['qk_rope_head_dim']
                                            ,bias=False ,)
        self.kv_a_layernorm = RMSNorm(model_args['kv_lora_rank'])
        self.kv_b_proj = nn.Linear(model_args['kv_lora_rank'] ,self.num_heads * (self.q_head_dim - self.qk_rope_head_dim +
                                                                             self.v_head_dim) ,bias=False ,)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim ,d_model, bias=False ,)

        self.apply_res = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))    #q
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)   #qc, qr

        compressed_kv = self.kv_a_proj_with_mqa(x)     #c
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)     #kc, kr

        k_pe = k_pe.view(batch, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
              .view(batch, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
              .transpose(1, 2))
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_pe, k_pe = apply_rope(q_pe, k_pe, freqs_cis)

        k_pe = k_pe.transpose(2, 1)
        q_pe = q_pe.transpose(2, 1)

        query_states = torch.cat((q_pe, q_nope), dim=-1)
        key_states = torch.cat((k_pe, k_nope), dim=-1)

        attn_mtx = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        mask = mask[:, 150:] if mask.size(1) > 150 else mask           # when applied to LM1b tasks, changes are required
        attn_mtx = attn_mtx.masked_fill(mask[:seq_len, :seq_len].reshape(1, 1, seq_len, seq_len) == 1, float('-inf'))
        attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(key_states)
        attn_mtx = self.attn_dropout(attn_mtx)
        output = torch.matmul(attn_mtx, value_states)  # (batch, n_head, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.v_head_dim)
        # final projection into the residual stream
        output = self.o_proj(output)
        if self.apply_res:
            output = self.res_dropout(output) + x
        else:
            output = self.res_dropout(output)
            self.apply_res = True

        return output