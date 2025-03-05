import time

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from layers2D3 import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
import torch.nn.functional as F
from itertools import chain
import sys
from torch import profiler
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch.nn.init as init
import time

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
    seq_ids_flattened = seq_ids.view(-1, k)

    seq_ids, _ = torch.sort(seq_ids, dim=2, descending=False)
    ids = batch_ids.to(device) * indices_len + seq_ids_flattened.to(device)
    ids = ids.flatten()

    return values, ids, seq_ids

def preprocess(X, route_mat):
    route_mat = route_mat.transpose(1, 2)
    batch_size, seq_len, d_model = X.shape
    n_expert = route_mat.shape[1]

    input_ = X.view(batch_size * seq_len, d_model)

    route_flatten = route_mat.reshape(-1, seq_len)

    max_len = torch.max(torch.sum(route_flatten > 0, dim=1))  #此前已将无需路由的块的索引全部置为0

    max_len = min(max_len, int(seq_len * 0.6))

    route_mat = route_mat.transpose(0, 1)

    _, ids, seq_ids = topk_indices(route_mat, max_len)

    x = torch.index_select(input_, 0, ids)

    x = x.view(n_expert, batch_size, max_len, d_model)          #!!!这里可能有问题,先检查 一下维度
    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.cuda()

class SwitchGate(nn.Module):
    def __init__(self, n_head, d_model, config):
        super().__init__()
        #注意：常规设置：patch_size = embed_dim

        self.dim = d_model
        self.num_experts = n_head
        self.capacity_factor = config['capacity']
        self.epsilon = config['epsilon']
        self.w_gate = nn.Linear(self.dim, self.num_experts)
        # self.topk = config['attn_topk']  ##!!!根据情况选择是使用topk还是transformer_topk
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
        gate_scores = F.softmax(X, dim=-1)    #token路由版本

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

        if use_aux_loss:
            pass

        return gate_scores, None

class SparseAttention(nn.Module):
    def __init__(self, d_head, n_experts, dropout, tgt_len):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=dropout)
        self.head_dim = d_head
        self.n_experts = n_experts
        self.tgt_len = tgt_len

    def forward(self, Q, K, V, route_mat, ids, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, route_mat, keep_shape=False)

        dot = scores / math.sqrt(self.head_dim)

        mask = mask[:, self.tgt_len:] if mask.size(1) > self.tgt_len else mask

        # The following is the normal processing method for masks

        # mask0 = mask
        # #ids的shape: (n_head, bhs, max_len)
        # ids1 = ids.unsqueeze(3).expand(-1, -1, -1, mask.shape[1])
        # mask = mask.squeeze(2).expand(ids.shape[0], ids.shape[1], mask.shape[0], mask.shape[1])
        # mask = torch.gather(mask, dim=2, index=ids1)
        # ids2 = ids.unsqueeze(2).expand(-1, -1, mask.shape[-2], -1)
        # mask = torch.gather(mask, dim=3, index=ids2)

        #However, due to the symmetric mask index during the retrieval process, the final result is still an upper triangular matrix
        #The following is an equivalent simplification method for masks (simplified but with the same effect)

        mask = mask[:dot.size(2), :dot.size(3)]
        dot.masked_fill_(mask[None, None, :, :].squeeze(-1).bool(), -float('inf'))
        dot = nn.functional.softmax(dot, dim=-1)
        scores = self.drop_attn(dot)
        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, route_mat, keep_shape=False,
                                                    combine_head=False)

        return X

class SparseAttention1(nn.Module):                            #Non parallel version
    def __init__(self, d_head, n_experts, dropout):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=dropout)
        self.head_dim = d_head
        self.n_experts = n_experts


    def forward(self, Q, K, V, route_mat, ids, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, route_mat, keep_shape=False)

        for i in range(len(scores)):
            if (len(scores[i]) != 0):
                dot = scores[i] / math.sqrt(self.head_dim)
                dot = dot - 1e6 * (1 - mask[ids[i], None, None, :])
                dot = nn.functional.softmax(dot, dim=-1)
                scores[i] = self.drop_attn(dot)

        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, route_mat, keep_shape=False,
                                                    combine_head=False)

        return X

class SparseSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, n_head, d_model, d_head, dropout, config, pre_lnorm=False, **kwargs):
        super(SparseSelfAttention, self).__init__()

        self.grad_checkpointing = False
        self.dim = d_model
        self.num_head = n_head
        self.head_dim = d_head
        self.n_experts = n_head

        self.switch_gate = SwitchGate(n_head, d_model, config)

        self.W_qkv = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_head * self.head_dim * 3,
            world_size=self.n_experts,
            gather_output=True,
            init_method=config['init_method'])

        self.dconv_fc = None
        self.attn = SparseAttention(d_head, self.n_experts, dropout, config['tgt_len'])

        self.drop_attn = torch.nn.Dropout(p=dropout)

        self.ff = RowParallelLinearWithMoE(
            self.num_head * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=config['output_layer_init_method'],
            skip_bias_add=False)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, X, attn_mask=None, mems=None):
        batch_size, seq_len, d_model = X.shape
        Y = X
        route_mat, aux_loss = self.switch_gate(X, use_aux_loss=True)
        X, ids, seq_ids = preprocess(X, route_mat)

        QKV, _ = self.W_qkv(X)#, ids, keep_shape=False, split_head=False)

        QKV = QKV.view(self.n_experts, batch_size, -1, 3, d_model // self.n_experts)

        Q = QKV[:, :, :, 0, :]
        K = QKV[:, :, :, 1, :]
        V = QKV[:, :, :, 2, :]

        attn_out = self.attn(Q, K, V, route_mat, seq_ids, attn_mask)    ##!!!!!!!!!!mask可能也要分开

        outputs, _ = self.ff(attn_out, keep_shape=False)

        if (len(outputs.shape) == 3):
            outputs = outputs.view(self.n_experts, batch_size, -1, outputs.shape[2])

        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(Y)
        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        if self.pre_lnorm:
            ##### residual connection
            output = Y + outs
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(Y + outs)

        return output
