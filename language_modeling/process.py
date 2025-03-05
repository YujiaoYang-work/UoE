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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

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

    max_len = torch.max(torch.sum(route_flatten > 0, dim=1))

    route_mat = route_mat.transpose(0, 1)

    _, ids, seq_ids = topk_indices(route_mat, max_len)
    # print(n_expert, batch_size, max_len, d_model, "fvmdfjmvdjfivsd", X.shape, ids.max())
    x = torch.index_select(input_, 0, ids)

    x = x.view(n_expert, batch_size, max_len, d_model)
    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.cuda()

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
        #注意：常规设置：patch_size = embed_dim

        self.dim = d_model
        self.num_experts = n_head
        self.capacity_factor = config['capacity']
        self.epsilon = config['epsilon']
        self.w_gate = nn.Linear(self.dim, self.num_experts)
        # self.s_gate = nn.Linear(config.max_seq_len, config.max_seq_len // patch_size)
        # self.topk = config['attn_topk']  ##!!!根据情况选择是使用topk还是transformer_topk
        self.topk = n_head  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意！！！

    def forward(self, X, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        batch_size, seq_len, d_model = X.shape

        # with profiler.profile(use_cuda=True, record_shapes=False, profile_memory=True) as prof:

        X = self.w_gate(X)
        gate_scores = F.softmax(X, dim=-1)    #token路由版本

        # prof.export_chrome_trace("/home/yujiao/a1/profile.json")
        # gate_scores = F.softmax(self.s_gate(self.w_gate(X).transpose(1, 2)), dim=-1).transpose(1, 2)    #patch路由版本
        # print(gate_scores, "gate1")
        #输出：[batch_size, seq_len, n_experts] (token路由版本)
        #输出：[batch_size, patch_len, n_experts] (patch路由版本)
        # print(self.topk)
        top_k_scores, top_k_indices = gate_scores.topk(self.topk, dim=-1)

        # # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * batch_size)
        # top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # print(top_k_indices.shape, "idxsfdffs",gate_scores.shape)
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
            pass       #暂时忽略
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts
            #
            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()
            #
            # return gate_scores, loss

        return gate_scores, None