from model.layer.layers_v2 import ColumnParallelLinearWithMoE, RowParallelLinearWithMoE
from process.process_v2 import SwitchGate, preprocess
from model.layer.linear import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class U_MLP(nn.Module):
    def __init__(self, config):
        super(U_MLP, self).__init__()

        self.dim = config.transformer_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.sub_dim = config.transformer_dim // config.n_experts
        self.sub_hidden_dim = config.transformer_hidden_dim // config.n_experts

        self.n_experts = config.n_experts
        self.topk = config.topk

        self.switch_gate = SwitchGate(config, is_attn=False)

        self.linear1 = ColumnParallelLinearWithMoE(
            config.transformer_dim,
            config.transformer_hidden_dim,
            config.n_experts,
            gather_output=False,
            init_method=config.init_method,
            skip_bias_add=False)

        self.linear2 = ColumnParallelLinearWithMoE(
            config.transformer_dim,
            config.transformer_hidden_dim,
            config.n_experts,
            gather_output=False,
            init_method=config.init_method,
            skip_bias_add=False)

        self.linear3 = RowParallelLinearWithMoE(
            config.transformer_hidden_dim,
            config.transformer_dim,
            config.n_experts,
            init_method=config.init_method,
            skip_bias_add=False)

        self.drop = nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        y = x
        bhs, seq_len, d_model = x.shape
        route_weight = self.switch_gate(x)
        x, ids, seq_ids, _ = preprocess(x, route_weight, None, None)

        x = F.silu(self.linear1(x)) * self.linear2(x)

        outputs = self.linear3(x)

        outputs = outputs.view(outputs.shape[0], bhs, -1, outputs.shape[-1])
        seq_ids = seq_ids.unsqueeze(3).expand(outputs.shape)
        outs = torch.zeros_like(y)
        for i in range(self.n_experts):
            outs = outs.scatter_add(1, seq_ids[i], outputs[i])

        outs = outs * torch.sum(route_weight, dim=-1, keepdim=True)
        outs = self.drop(outs)

        return outs