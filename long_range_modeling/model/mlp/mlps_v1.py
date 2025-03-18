import torch
import torch.nn as nn
from model.layer.layers_v1 import split_tensor_along_last_dim, ColumnParallelLinearWithMoE, RowParallelLinearWithMoE
from process.process_v1 import preprocess, preprocess2, preprocessMH
from itertools import chain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x0 = x[:, :2048]
        batch_size, seq_len, d_model = x.shape
        y = x

        X_flattened = x0.reshape(batch_size, -1)
        route_prob = self.switch(X_flattened).view(batch_size, self.n_experts)
        route_prob = nn.functional.softmax(route_prob, dim=-1)  # .transpose(0, 1)

        x, idx_list = preprocess2(x, route_prob, k=self.topk)
        output_list = []
        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                output_list.append(self.linear1[i](x[i]))
            else:
                output_list.append([])

        output = torch.zeros([batch_size, seq_len, self.hidden_dim]).to(device)
        for i in range(self.n_experts):
            if (len(idx_list[i]) != 0):
                output[:, :,  (i * self.sub_hidden_dim) : ((i+1) * self.sub_hidden_dim)].index_add_(0, torch.tensor(idx_list[i]), output_list[i])

        x = self.act1(output)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x

class U_MLP(nn.Module):
    def __init__(self, config):
        super(U_MLP, self).__init__()

        self.dim = config.transformer_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.sub_dim = config.transformer_dim // config.n_experts
        self.sub_hidden_dim = config.transformer_hidden_dim // config.n_experts

        self.n_experts = config.n_experts
        self.topk = config.topk

        self.switch = nn.Linear(self.dim * config.max_seq_len, self.n_experts)

        self.linear1 = ColumnParallelLinearWithMoE(
            self.dim,
            self.hidden_dim,
            self.n_experts,
            gather_output=False,
            init_method=config.init_method,
            skip_bias_add=False)

        self.act1 = nn.ModuleList()
        for i in range(self.n_experts):
            self.act1.append(nn.GELU())
        self.drop1 = nn.ModuleList()
        for i in range(self.n_experts):
            self.drop1.append(torch.nn.Dropout(p=config.dropout_prob))

        self.linear2 = RowParallelLinearWithMoE(
            self.hidden_dim,
            self.dim,
            self.n_experts,
            init_method=config.init_method,
            skip_bias_add=False)

        self.drop2 = torch.nn.Dropout(p=config.dropout_prob)
        self.linear3 = nn.Linear(self.dim, self.dim)
        self.drop3 = torch.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
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
        route_prob = nn.functional.softmax(route_prob, dim=-1)

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