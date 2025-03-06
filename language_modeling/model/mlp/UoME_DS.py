import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer.layers_v2 import split_tensor_along_last_dim, ColumnParallelLinearWithMoE, RowParallelLinearWithMoE
from itertools import chain


def topk_indices(route_prob, k):
    n_experts, batch_size, indices_len = route_prob.shape
    values, seq_ids = torch.topk(route_prob, k, dim=-1)

    batch_ids = torch.arange(batch_size).view(1, batch_size, 1).expand(n_experts, batch_size, k).reshape(-1, k)

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

    max_len = max(torch.sum(route_flatten > 0, dim=1))
    # aux_loss = ((seq_len - max_len * 0.5) ** 2) * 0.001

    max_len = int(seq_len * 0.5)

    route_mat = route_mat.transpose(0, 1)
    _, ids, seq_ids = topk_indices(route_mat, max_len)
    x = torch.index_select(input_, 0, ids)
    x = x.view(n_expert, batch_size, max_len, d_model)

    ids = ids.view(n_expert, batch_size, max_len)

    return x, ids, seq_ids.to(x.device)

class SwitchGate(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        #注意：常规设置：patch_size = embed_dim

        self.dim = d_model
        self.num_experts = config['n_experts']
        self.capacity_factor = config['capacity']
        self.epsilon = config['epsilon']
        self.w_gate = nn.Linear(self.dim, self.num_experts)
        self.topk = config['topk']

    def forward(self, X, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        batch_size, seq_len, d_model = X.shape
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(X), dim=-1)

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

        return gate_scores

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.mlpblock = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, x):
        return self.mlpblock(x)

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(MLP2, self).__init__()
        self.mlpblock = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, output_dim),
            torch.nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.mlpblock(x)


class MH_MLP(nn.Module):
    def __init__(self, config):
        super(MH_MLP, self).__init__()

        sub_dim = config.transformer_dim // config.n_experts
        sub_hidden_dim = config.transformer_hidden_dim // config.n_experts

        self.n_experts = config.n_experts
        self.linear1 = nn.ModuleList()
        for i in range(config.n_experts):
            self.linear1.append(nn.Linear(sub_dim, sub_hidden_dim))
        self.act1 = nn.GELU()
        self.drop1 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear2 = nn.Linear(config.transformer_hidden_dim, config.transformer_dim)
        self.drop2 = torch.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        x = split_tensor_along_last_dim(x, self.n_experts)
        output_list = []
        for i in range(self.n_experts):
            output_list.append(self.linear1[i](x[i]))
        x = torch.cat(output_list, dim=-1)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x

class S_MH_MLP(nn.Module):
    def __init__(self, config):
        super(S_MH_MLP, self).__init__()

        self.dim = config.transformer_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.sub_dim = config.transformer_dim // config.n_experts
        self.sub_hidden_dim = config.transformer_hidden_dim // config.n_experts

        self.n_experts = config.n_experts
        self.topk = config.topk

        self.switch = nn.Linear(config.transformer_dim * config.max_seq_len, self.n_experts)

        self.linear1 = nn.ModuleList()
        for i in range(config.n_experts):
            self.linear1.append(MLP2(self.sub_dim, self.sub_hidden_dim, self.sub_hidden_dim, config.dropout_prob))
        self.act1 = nn.GELU()
        self.drop1 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear2 = nn.Linear(config.transformer_hidden_dim, config.transformer_dim)
        self.drop2 = torch.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        y = x

        X_flattened = x.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)  # .transpose(0, 1)

        x, idx_list = preprocess2(x, route_prob, k=self.topk)
        output_list = []
        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                output_list.append(self.linear1[i](x[i]))
            else:
                output_list.append([])

        output = torch.zeros([batch_size, seq_len, self.hidden_dim]).to(y.device)
        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                output[:, :,  (i * self.sub_hidden_dim) : ((i+1) * self.sub_hidden_dim)].index_add_(0, torch.tensor(idx_list[i]), output_list[i])

        x = self.act1(output)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x

class U_MLP_v1(nn.Module):                 #Non parallel version
    def __init__(self, config):
        super(U_MLP, self).__init__()

        self.dim = config.transformer_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.sub_dim = config.transformer_dim // config.n_experts
        self.sub_hidden_dim = config.transformer_hidden_dim // config.n_experts

        self.n_experts = config.n_experts
        self.topk = config.topk

        self.vis = vis
        self.grad_checkpointing = False
        self.dim = config.hidden_size
        self.num_head = config.transformer["num_heads"]
        self.head_dim = config.hidden_size // config.transformer["num_heads"]
        # self.attn_type = config.attn_type

        self.n_experts = config.transformer["num_heads"]

        self.switch = nn.Linear(config.transformer_dim * config.max_seq_len, self.n_experts)

        self.linear1 = ColumnParallelLinearWithMoE(
            config.transformer_dim,
            config.transformer_hidden_dim,
            config.n_experts,
            gather_output=False,
            init_method=config.init_method,
            skip_bias_add=False)

        self.act1 = nn.ModuleList()
        for i in range(config.n_experts):
            self.act1.append(nn.GELU())
        self.drop1 = nn.ModuleList()
        for i in range(config.n_experts):
            self.drop1.append(torch.nn.Dropout(p=config.dropout_prob))

        self.linear2 = RowParallelLinearWithMoE(
            config.transformer_hidden_dim,
            config.transformer_dim,
            config.n_experts,
            init_method=config.init_method,
            skip_bias_add=False)

        self.drop2 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear3 = nn.Linear(config.transformer_dim, config.transformer_dim)
        self.drop3 = torch.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        # x0 = x[:, :4096, :]

        batch_size, seq_len, d_model = x.shape
        y = x

        X_flattened = x.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)
        x, idx_list = preprocess(x, route_prob, k=self.topk)
        flattened_list = list(chain(*idx_list))

        x, _ = self.linear1(x, idx_list, keep_shape=False, split_head=False)

        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                x[i] = self.drop1[i](self.act1[i](x[i]))

        output_list, _ = self.linear2(x, idx_list=idx_list, keep_shape=False)

        filtered_list = [sublist for sublist in output_list if (len(sublist) > 0)]
        output_gathered = torch.cat(filtered_list, dim=0)
        output = torch.zeros_like(y)
        output.index_add_(0, torch.tensor(flattened_list).cuda(), output_gathered)

        output = self.drop2(output)

        # output = y + output
        # x = self.linear3(output)
        # x = self.drop3(x)

        return output

class U_MLP(nn.Module):
    def __init__(self, d_model, d_inner, dropout, config, pre_lnorm):
        super(U_MLP, self).__init__()

        self.grad_checkpointing = False
        self.dim = d_model
        self.hidden_dim = d_inner
        self.n_experts = config['n_experts']

        self.switch_gate = SwitchGate(d_model, config)

        self.linear1 = ColumnParallelLinearWithMoE(
            self.dim,
            self.hidden_dim,
            self.n_experts,
            gather_output=False,
            init_method=config['init_method'],
            skip_bias_add=False)

        self.act1 = nn.GELU()
        self.drop1 = torch.nn.Dropout(p=dropout)

        self.linear2 = RowParallelLinearWithMoE(
            self.hidden_dim,
            self.dim,
            self.n_experts,
            init_method=config['init_method'],
            skip_bias_add=False)

        self.drop2 = torch.nn.Dropout(p=config["dropout_rate"])
        self.layer_norm = nn.LayerNorm(d_model)

        self.linear3 = nn.Linear(self.dim, self.dim)
        self.drop3 = torch.nn.Dropout(p=config["dropout_rate"])

        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        bhs, seq_len, d_model = x.shape
        y = x

        route_mat = self.switch_gate(x, use_aux_loss=True)
        x, ids, seq_ids = preprocess(x, route_mat)

        x, _ = self.linear1(x)#, route_mat, keep_shape=False, split_head=False)

        # print(x.shape, "Fgkfjgoib", y.shape)
        x = self.drop1(self.act1(x))

        outputs, _ = self.linear2(x)#, keep_shape=False)

        if (len(outputs.shape) == 3):
            outputs = outputs.view(self.n_experts, bhs, -1, outputs.shape[2])

        # outputs = torch.stack(outputs, dim=0)
        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(y)

        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        outs = self.drop2(outs)

        if self.pre_lnorm:
            ##### residual connection
            output = y + outs
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(y + outs)

        return output

class MH_U_MLP(nn.Module):
    def __init__(self, config):
        super(MH_U_MLP, self).__init__()

        self.dim = config.transformer_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.sub_dim = config.transformer_dim // config.n_experts
        self.sub_hidden_dim = config.transformer_hidden_dim // config.n_experts
        self.n_heads = config.num_heads
        self.head_dim = config.transformer_dim // config.num_heads
        self.head_hidden_dim = config.transformer_hidden_dim // config.num_heads

        self.n_experts = config.n_experts
        self.topk = config.topk

        self.switch = nn.Linear(config.transformer_dim * config.max_seq_len, self.n_experts)

        self.linear1 = ColumnParallelLinearWithMoE(
            self.head_dim,
            self.head_hidden_dim,
            config.n_experts,
            gather_output=False,
            init_method=config.init_method,
            skip_bias_add=False)

        self.act1 = nn.GELU()
        self.drop1 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear2 = RowParallelLinearWithMoE(
            self.head_hidden_dim,
            self.head_dim,
            config.n_experts,
            init_method=config.init_method,
            skip_bias_add=False)

        self.drop2 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear3 = nn.Linear(config.transformer_hidden_dim, config.transformer_dim)
        self.drop3 = torch.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        x0 = x[:, :2048]

        batch_size, seq_len, d_model = x.shape
        y = x

        X_flattened = x0.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)  # .transpose(0, 1)

        x = self.split_heads(x)

        x, idx_list = preprocessMH(x, route_prob, k=self.topk)
        flattened_list = list(chain(*idx_list))

        x, _ = self.linear1(x, idx_list, keep_shape=False, split_head=False)

        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                x[i] = self.drop1(self.act1(x[i]))

        output_list, _ = self.linear2(x, idx_list=idx_list, keep_shape=False)

        filtered_list = [sublist for sublist in output_list if (len(sublist) > 0)]
        output_gathered = torch.cat(filtered_list, dim=0)
        output_gathered = self.combine_heads(output_gathered)
        output = torch.zeros_like(y)

        output.index_add_(0, torch.tensor(flattened_list).cuda(), output_gathered)

        output = y + output

        x = self.linear3(output)
        x = self.drop3(x)

        return x

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.n_heads * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.n_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X
