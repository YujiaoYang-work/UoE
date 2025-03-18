import torch
import torch.nn as nn
import math
from model.layer.layers_v3 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
from process.process_v2 import SwitchGate, preprocess, RMSNorm, apply_rotary_emb, precompute_freqs_cis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, config):
        super(SparseSelfAttention, self).__init__()

        self.grad_checkpointing = config.attention_grad_checkpointing
        self.dim = config.transformer_dim
        self.num_head = config.attn_n_experts
        self.head_dim = config.transformer_dim // self.num_head
        self.attn_type = config.attn_type
        self.n_experts = config.attn_n_experts

        self.use_norm = False
        self.apply_rope = True

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

        self.kv_norm = RMSNorm(self.head_dim)

        self.attn = SparseAttention(config)

        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)

        self.ff = RowParallelLinearWithMoE(
            self.num_head * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=config.output_layer_init_method,
            skip_bias_add=False)

        self.layer_norm = nn.LayerNorm(self.dim)

        self.register_buffer("freqs_cis", precompute_freqs_cis(self.qk_rope_head_dim, seqlen=config.max_seq_len), persistent=False)


    def forward(self, x, mask=None):

        batch_size, seq_len, d_model = x.shape
        y = x

        freqs_cis = self.freqs_cis
        if self.apply_rope:
            x = x.view(batch_size, seq_len, self.n_experts, -1)
            x_nope, x_pe = torch.split(x, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            x_pe = apply_rotary_emb(x_pe, freqs_cis)
            x = torch.cat((x_pe, x_nope), dim=-1).view(batch_size, seq_len, d_model)

        route_mat = self.switch_gate(x)

        x, ids, seq_ids, mask = preprocess(x, route_mat, mask, freqs_cis)

        qkv = self.W_qkv(x)
        qkv = qkv.view(self.n_experts, batch_size, -1, 3, d_model // self.n_experts)

        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]

        if self.use_norm:
            k = self.kv_norm(k)
            v = self.kv_norm(v)

        attn_out = self.attn(q, k, v, route_mat, mask)

        outputs = self.ff(attn_out, keep_shape=False)

        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(y)
        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        return outs