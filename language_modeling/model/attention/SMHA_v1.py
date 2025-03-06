import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from model.model import RowParallelLinearWithMoE, RowParallelMatmulWithMoE, ColumnParallelMatmulWithMoE, ColumnParallelLinearWithMoE
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

import torch.nn.init as init

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

def repeat_tensor(tensor, exp_dim=128):
    dim1, dim2, dim3 = tensor.shape
    repeated_tensor = tensor.unsqueeze(-1).repeat(1, 1, 1, exp_dim).view(dim1, dim2, dim3*exp_dim)

    return repeated_tensor

def split_heads(X, num_head, head_dim):
    X = X.reshape(X.size(0), X.size(1), num_head, head_dim)
    X = X.transpose(1, 2)
    return X

def combine_heads(X, num_head, head_dim):
    X = X.transpose(1, 2)
    X = X.reshape(X.size(0), X.size(1), num_head * head_dim)
    return X

class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=0.1)
        self.head_dim = config.hidden_size // config.transformer["num_heads"]
        self.n_experts = config.hidden_size // 128

    def forward(self, Q, K, V, idx_list, mask):

        scores = RowParallelMatmulWithMoE(Q, K, self.n_experts, idx_list, keep_shape=False)

        for i in range(len(scores)):
            if (len(scores[i]) != 0):
                dot = scores[i] / math.sqrt(self.head_dim)
                dot = dot - 1e6 * (1 - mask[route_mat[i], None, None, :])
                dot = nn.functional.softmax(dot, dim=-1)
                scores[i] = self.drop_attn(dot)

        X = ColumnParallelMatmulWithMoE(scores, V, self.n_experts, idx_list, keep_shape=False,
                                                    combine_head=False)

        return X, None

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

def topk_indices(route_prob, k):
    batch_size, n_experts = route_prob.shape

    values, seq_ids = torch.topk(route_prob, k, dim=-1)
    batch_ids = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, k).reshape(-1, k)
    seq_ids, _ = torch.sort(seq_ids, dim=1, descending=False)
    ids = batch_ids.to(device) * n_experts + seq_ids.to(device)

    return values, ids, seq_ids

def get_reverse_ids(all_ids, seq_ids):
    all_ids = all_ids.tolist()
    seq_ids = seq_ids.tolist()
    reverse_ids = []
    for i in range(len(all_ids)):
        set1 = set(all_ids[i])
        set2 = set(seq_ids[i])
        reverse_set = set1.difference(set2)
        reverse_ids.append(list(reverse_set))
    reverse_ids = torch.tensor(reverse_ids)

    return reverse_ids
def process_X(X, ids, n_expert):
    idx_list = [torch.eq(ids, i).nonzero(as_tuple=True)[0] for i in range(n_expert)]
    flattened_list = list(chain(*idx_list))
    X = torch.index_select(X, 0, torch.tensor(flattened_list).cuda())
    sub_lengths = [len(sublist) for sublist in idx_list]
    X = torch.split(X, sub_lengths, dim=0)
    X = [sub_tensor if sub_length != 0 else [] for sub_tensor, sub_length in zip(X, sub_lengths)]

    return X, idx_list

def preprocess(X, route_prob, k=2):

    batch_size, n_expert = route_prob.shape
    values, seq_ids = torch.topk(route_prob, k, dim=-1)
    X_, idx_list = process_X(X, seq_ids, n_expert)

    return X_, idx_list

class SparseSelfAttention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()

        self.grad_checkpointing = False

        self.vis = vis
        self.dim = config.hidden_size
        self.head_dim = 128
        self.num_head = config.transformer["num_heads"]
        self.topk = 4
        # self.num_head = 1

        self.n_experts = config.hidden_size // self.head_dim

        # self.n_patches = config.max_seq_len // self.head_dim
        self.n_sub_dim = self.n_experts = self.num_head

        self.switch = nn.Linear(config.hidden_size * 197, self.n_experts)

        self.W_q = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_head * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=init.xavier_normal_)

        self.W_k = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_head * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=init.xavier_normal_)

        self.W_v = ColumnParallelLinearWithMoE(
            self.dim,
            self.num_head * self.head_dim,
            world_size=self.n_experts,
            gather_output=True,
            init_method=init.xavier_normal_)

        self.dconv_fc = None

        self.attn = SparseAttention(config)

        self.ff = RowParallelLinearWithMoE(
            self.num_head * self.head_dim,
            self.dim,
            world_size=self.n_experts,
            init_method=init.xavier_normal_,
            skip_bias_add=False)

    def transpose_for_scores(self, x): 
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, X):
        batch_size, seq_len, d_model = X.shape

        Y = X
        X_flattened = X.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)  # .transpose(0, 1)

        mask = None

        X, idx_list = preprocess(X, route_prob, k=self.topk)
        flattened_list = list(chain(*idx_list))

        Q, _ = self.W_q(X, idx_list, keep_shape=False, split_head=False)
        K, _ = self.W_k(X, idx_list, keep_shape=False, split_head=False)
        V, _ = self.W_v(X, idx_list, keep_shape=False, split_head=False)

        if self.grad_checkpointing:
            attn_out = checkpoint(self.attn, Q, K, V, idx_list, mask.float())
        else:
            attn_out, attention_probs = self.attn(Q, K, V, idx_list, mask)

        weights = attention_probs if self.vis else None
        output_list, _ = self.ff(attn_out, idx_list=idx_list, keep_shape=False)
        filtered_list = [sublist for sublist in output_list if (len(sublist) > 0)]
        output_gathered = torch.cat(filtered_list, dim=0)
        output = torch.zeros_like(Y)
        output.index_add_(0, torch.tensor(flattened_list).cuda(), output_gathered)

        return output, weights
