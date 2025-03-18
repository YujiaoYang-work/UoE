"""
This file is from https://github.com/mlpen/Nystromformer
"""

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from model.layer.layers_v1 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
from itertools import chain
from process.process_v1 import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        self.head_dim = config.head_dim

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        # print(dot.shape, mask[:, None, None, :].shape)

        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        # print(dot.shape, "fvmpiobg")
        # sys.exit(0)

        attn = nn.functional.softmax(dot, dim=-1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, Q, K, V, mask):
        return V

class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.transformer_dim
        self.head_dim = config.transformer_dim // config.num_heads

        if (config.auto_group):
            self.n_experts = config.transformer_dim // config.expert_dim
        else:
            self.n_experts = config.attn_n_experts
        self.drop_attn = nn.ModuleList()

        for i in range(self.n_experts):
            self.drop_attn.append(torch.nn.Dropout(p=config.attention_dropout))

        self.batch_size = config.batch_size
        self.seq_len = config.max_seq_len
        self.topk = config.attn_topk

    def forward(self, Q, K, V, idx_list, mask, route_prob):

        # mask = mask[:, None, :]

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, idx_list, keep_shape=False)

        for i in range(len(scores)):
            if (len(scores[i]) != 0):
                dot = scores[i] / math.sqrt(self.dim)

                # dot = dot - 1e6 * (1 - mask[idx_list[i]]) #   ！！！！4.20：暂时弃用mask

                dot = nn.functional.softmax(dot, dim=-1)
                scores[i] = self.drop_attn[i](dot)

        flattened_list = list(chain(*idx_list))
        filtered_list = [sublist for sublist in scores if (len(sublist) > 0)]
        output_gathered = torch.cat(filtered_list, dim=0)
        attn_score = torch.zeros([self.batch_size, self.seq_len, self.seq_len], device=device)
        attn_score.index_add_(0, torch.tensor(flattened_list).cuda(), output_gathered)
        attn_score, idx_list = preprocess(attn_score, route_prob, k=self.topk)   #上面这些步骤能够进一步简化， 并且此时idx_list与前面的一样，因为用的是同一个route_prob

        X = ColumnParallelMatmulWithMoE(attn_score, V, self.n_experts, idx_list, keep_shape=False,
                                                    combine_head=False)

        return X

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

class SparseSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config.attention_grad_checkpointing

        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.topk = config.attn_topk
        # self.num_head = 1

        self.attn_type = config.attn_type

        if (config.auto_group):
            self.n_experts = config.transformer_dim // config.expert_dim
            self.n_patches = config.max_seq_len // config.expert_dim
            self.n_sub_dim = self.n_experts

        else:
            self.n_experts = config.attn_n_experts
            self.n_patches = config.n_patches

        self.switch = nn.Linear(config.transformer_dim * config.max_seq_len, self.n_experts)
        # self.switch = nn.Sequential(
        #     nn.Linear(config.transformer_dim*config.max_seq_len, self.n_experts*64),
        #     nn.GELU(),
        #     torch.nn.Dropout(p=config.dropout_prob),
        #     nn.Linear(self.n_experts*64, self.n_experts),
        #     torch.nn.Dropout(p=config.dropout_prob)
        # )

        # self.norm = nn.LayerNorm(self.n_experts*self.n_patches)

        # self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        # self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        # self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        #
        # self.W_q = ColumnParallelLinear(
        #     self.dim,
        #     self.num_head * self.head_dim,
        #     world_size=self.n_experts,
        #     gather_output=False,
        #     init_method=config.init_method)
        #
        # self.W_k = ColumnParallelLinear(
        #     self.dim,
        #     self.num_head * self.head_dim,
        #     world_size=self.n_experts,
        #     gather_output=False,
        #     init_method=config.init_method)
        #
        # self.W_v = ColumnParallelLinear(
        #     self.dim,
        #     self.num_head * self.head_dim,
        #     world_size=self.n_experts,
        #     gather_output=False,
        #     init_method=config.init_method)

        self.W_q = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_heads * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config.init_method)

        self.W_k = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_heads * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config.init_method)

        self.W_v = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_heads * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config.init_method)

        self.dconv_fc = None

        if self.attn_type == "softmax":
            # self.attn = SoftmaxAttention(config)
            self.attn = SparseAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type.startswith("linformer"):
            from attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)

        elif self.attn_type.startswith("reformer"):
            from attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type.startswith("nystrom"):
            from attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type.startswith("performer"):
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type.startswith("linear"):
            from attention_linear import LinearAttention
            self.attn = LinearAttention(config)
        else:
            # self.attn = SoftmaxAttention(config)
            self.attn = SparseAttention(config)

        # self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)
        self.ff = RowParallelLinearWithMoE(
            self.num_heads * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=config.output_layer_init_method,
            skip_bias_add=False)

    def forward(self, X, mask):
        ##算法中欠缺的部分：混合精度训练、mask的应用、多头机制的使用、cls_embed
        #如果要使用mask和混合精度，将代码中各自对应的语句取消注释即可（各自只有一句）
        #对于多头机制，首先要考察其合理性
        #首先需要将其跑起来然后再做更改，为此需要搞清楚函数调用链
        #已解决的问题：mask的应用、

        batch_size, seq_len, d_model = X.shape

        X_flattened = X.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)  # .transpose(0, 1)
        # with profiler.profile(use_cuda=True, record_shapes=False, profile_memory=True) as prof:

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        else:
            X, idx_list = preprocess(X, route_prob, k=self.topk)


            Q, _ = self.W_q(X, idx_list, keep_shape=False, split_head=False)
            K, _ = self.W_k(X, idx_list, keep_shape=False, split_head=False)
            V, _ = self.W_v(X, idx_list, keep_shape=False, split_head=False)

            # Q = self.split_heads(Q)
            # K = self.split_heads(K)
            # V = self.split_heads(V)

            # with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, Q, K, V, idx_list, mask.float())
            else:
                attn_out = self.attn(Q, K, V, idx_list, mask.float(), route_prob)
            # attn_out = self.combine_heads(attn_out)

        output_list, _ = self.ff(attn_out, idx_list=idx_list, keep_shape=False)

        flattened_list = list(chain(*idx_list))
        filtered_list = [sublist for sublist in output_list if (len(sublist) > 0)]
        output_gathered = torch.cat(filtered_list, dim=0)
        output = torch.zeros([batch_size, seq_len, d_model], device=device)
        output.index_add_(0, torch.tensor(flattened_list).cuda(), output_gathered)

        return output

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config.attention_grad_checkpointing

        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.num_head = config.num_heads

        self.attn_type = config.attn_type

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.dconv_fc = None

        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type.startswith("linformer"):
            from attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)

        elif self.attn_type.startswith("reformer"):
            from attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type.startswith("nystrom"):
            from attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type.startswith("performer"):
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type.startswith("linear"):
            from attention_linear import LinearAttention
            self.attn = LinearAttention(config)

        self.attn = SoftmaxAttention(config)
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):
        # with profiler.profile(use_cuda=True, record_shapes=False, profile_memory=True) as prof:

        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            V = self.split_heads(self.W_v(X))

            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
