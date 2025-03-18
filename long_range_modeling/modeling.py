import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.attention.attention import Attention

from model.attention.sparse_attention_v3 import SparseSelfAttention
from model.mlp.mlps_v2 import U_MLP

from model.baseline.moe import MoE

from model.baseline.DeepSeekV3 import Transformer as DeepSeekV3Transformer
from model.baseline.DeepSeekV3 import ModelArgs as DeepSeekV3ModelArgs

from model.baseline.config_sky import SkyworkMoeConfig
from model.baseline.SkyworkMoE import SkyworkDecoderLayer

from megatron.model.transformer import SwitchMLP, ParallelMLP, SparseMLP, MoE
from megatron.arguments import core_transformer_config_from_args


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_dim == config.transformer_dim

        self.dim = config.embedding_dim

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02)

        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embedding_dim)
        torch.nn.init.normal_(self.position_embeddings.weight, std=0.02)

        if config.debug:
            self.word_embeddings.weight[-1].data[:] = 0
            self.position_embeddings.weight[0].data[:] = 0

        self.dropout = torch.nn.Dropout(p=config.dropout_prob)

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device=device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.transformer_dim)
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p=config.dropout_prob)
        self.norm2 = nn.LayerNorm(config.transformer_dim)
        self.debug = config.debug

        self.mlpblock = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, X, mask, cls_embed=None):
        if cls_embed is None:
            X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        else:
            if cls_embed.shape[0] == 1:
                cls_embed = cls_embed.expand(X.shape[0], -1, -1)
            X_prepend = torch.cat([cls_embed, X], dim=1)
            if self.debug:
                cls_embed = self.norm1(cls_embed)
            X = self.dropout1(self.mha(self.norm1(X), mask, cls_embed)) + X_prepend
        X = self.mlpblock(self.norm2(X)) + X
        return X

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.transformer_dim)
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p=config.dropout_prob)
        self.norm2 = nn.LayerNorm(config.transformer_dim)
        self.debug = config.debug

        self.mlpblock = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

    def forward(self, X, mask, cls_embed=None):
        X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class ParallelTransformer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.norm1 = RMSNorm(config.transformer_dim)
        self.mha = SparseSelfAttention(config)
        self.dropout1 = torch.nn.Dropout(p=config.dropout_prob)
        self.norm2 = RMSNorm(config.transformer_dim)
        self.debug = config.debug

        self.mlpblock = U_MLP(config)

    def forward(self, X, mask, cls_embed=None):
        cls_embed = None
        if cls_embed is None:
            X = self.dropout1(self.mha(self.norm1(X), mask)) + X
        else:
            if cls_embed.shape[0] == 1:
                cls_embed = cls_embed.expand(X.shape[0], -1, -1)
            X_prepend = torch.cat([cls_embed, X], dim=1)
            if self.debug:
                cls_embed = self.norm1(cls_embed)
            X = self.dropout1(self.mha(self.norm1(X), mask, cls_embed)) + X_prepend

        X = self.mlpblock(self.norm2(X)) + X

        return X

class SkyDecoderLayer(nn.Module):
    def __init__(self, idx, config):
        super(SkyDecoderLayer, self).__init__()
        config = SkyworkMoeConfig(hidden_size=config.transformer_dim, intermediate_size=config.transformer_hidden_dim // config.n_experts,
        num_attention_heads=config.attn_n_experts, max_position_embeddings=config.transformer_dim // 2, num_experts=[config.n_experts])
        self.layer = SkyworkDecoderLayer(config, idx)

    def forward(self, dec_inp, mask=None, cls_embed=None):
        output = self.layer(dec_inp, attention_mask=mask)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None):
        c = h
        head_q = self.q_net(h)

        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        attn_score = torch.einsum('ibnd,jbnd->ijbn', head_q, head_k)

        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(~attn_mask.transpose(0, 1)[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(~attn_mask[:,:,:,None].bool(), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, head_v)
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return attn_out

class XMoETransformerLayer(nn.Module):
    def __init__(self, args, config, n_head, d_model, d_head, d_inner, dropout):
        super(XMoETransformerLayer, self).__init__()

        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # Self attention.
        self.self_attention = MultiHeadAttn(n_head, d_model, d_head, dropout)

        self.hidden_dropout = dropout

        self.apply_residual_connection_post_layernorm \
            = False

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # MLP
        self.num_experts = 4
        if args.num_experts_switch is not None:
            self.mlp = SwitchMLP(config)  # Megatron-LM's MoE
        else:
            if self.num_experts <= 1:  # dense, not MoE
                self.mlp = ParallelMLP(config)
            else:  # DeepSpeed's MoE
                enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism

                if args.sparse_mlp:
                    self.mlp = SparseMLP(args, args.hidden_size,
                                         ParallelMLP(args, config,
                                                     moe=True,
                                                     enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                                         # ExpertMLPs(args, config,
                                         #            moe=True,
                                         #            num_experts=self.num_experts,
                                         #            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                                         num_experts=self.num_experts,
                                         ep_size=args.moe_expert_parallel_size,
                                         k=args.topk,
                                         use_residual=(args.mlp_type == 'residual'),
                                         capacity_factor=args.moe_train_capacity_factor,
                                         eval_capacity_factor=args.moe_eval_capacity_factor,
                                         min_capacity=args.moe_min_capacity,
                                         drop_tokens=args.moe_token_dropping, use_tutel=args.use_tutel,
                                         enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
                                         threshold=args.threshold,
                                         placeholder_expert=args.placeholder_expert,
                                         view_num=args.gate_view_num,
                                         scale_moe=args.scale_moe)
                else:
                    print("yyh here using Deepspeed MoE!")
                    self.mlp = MoE(args.hidden_size,
                                   ParallelMLP(args, config,
                                               moe=True,
                                               enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                                   num_experts=self.num_experts,
                                   ep_size=args.moe_expert_parallel_size,
                                   k=args.topk,
                                   use_residual=(args.mlp_type == 'residual'),
                                   capacity_factor=args.moe_train_capacity_factor,
                                   eval_capacity_factor=args.moe_eval_capacity_factor,
                                   min_capacity=args.moe_min_capacity,
                                   drop_tokens=args.moe_token_dropping, use_tutel=args.use_tutel,
                                   enable_expert_tensor_parallelism=enable_expert_tensor_parallelism, )

        self.mlp = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=config.dropout_prob),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=config.dropout_prob)
        )

        self.args = args

    def forward(self, hidden_states, attention_mask=None, mems=None,
                enc_input_ids=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = \
            self.self_attention(layernorm_output, attention_mask, mems)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        out = torch.nn.functional.dropout(attention_output,
                                          p=self.hidden_dropout,
                                          training=self.training)
        layernorm_input = residual + out

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        my_probe = {}

        if self.num_experts == 1:
            mlp_output, mlp_bias, non_zero_ratio = self.mlp(layernorm_output)
            my_probe["non_zero_ratio"] = non_zero_ratio
        else:
            mlp_output, moe_loss, _, gate_info = self.mlp(layernorm_output, now_training_process=1,
                                                          enc_input_ids=enc_input_ids)

            my_probe = gate_info

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias
        out = torch.nn.functional.dropout(mlp_output,
                                          p=self.hidden_dropout,
                                          training=self.training)
        output = residual + out

        return output, moe_loss, my_probe

class XMoETransformerLayer(nn.Module):
    """A single XMoE transformer layer.
    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, args, config, n_head, d_model, d_head, d_inner, dropout):
        super(XMoETransformerLayer, self).__init__()

        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # Self attention.
        self.self_attention = MultiHeadAttn(n_head, d_model, d_head, dropout)

        self.hidden_dropout = dropout
        self.apply_residual_connection_post_layernorm \
            = False

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(p=0.1),
            nn.Linear(config.transformer_hidden_dim, config.transformer_dim),
            torch.nn.Dropout(p=0.1)
        )

        self.args = args
    def forward(self, hidden_states, attention_mask=None, mems=None,
                enc_input_ids=None):
        # hidden_states: [s, b, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = \
            self.self_attention(layernorm_output, attention_mask)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        out = torch.nn.functional.dropout(attention_output,
                                          p=self.hidden_dropout,
                                          training=self.training)
        layernorm_input = residual + out
        layernorm_input = layernorm_input.transpose(0, 1).contiguous()

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        output = self.mlp(layernorm_output)

        return output, None, None

class XMoEDecoderLayer(nn.Module):
    def __init__(self, args):
        super(XMoEDecoderLayer, self).__init__()

        config = core_transformer_config_from_args(args)
        self.n_experts = 4

        n_head = config.attn_n_experts
        d_model = config.transformer_dim
        d_head = d_model // n_head
        d_inner = config.transformer_hidden_dim
        dropout = config.attention_dropout

        self.layer = XMoETransformerLayer(args, config, n_head, d_model, d_head, d_inner, dropout)

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output, moe_loss, my_probe = self.layer(dec_inp, dec_attn_mask, mems)

        return output

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config.num_layers
        self.tied_weights = config.tied_weights
        self.cls_last_layer = config.cls_last_layer

        self.embeddings = Embeddings(config)

        if config.cls_token or self.cls_last_layer:
            self.cls_embed = nn.Parameter(torch.zeros(1, 1, config.transformer_dim))
        else:
            self.cls_embed = None

        deepseekv3_config = DeepSeekV3ModelArgs()

        for idx in range(self.num_layers):
            # setattr(self, f"transformer_{idx}", Transformer(config))                                           #Transformer
            setattr(self, f"transformer_{idx}", ParallelTransformer(config, deepseekv3_config))                  #UoE
            # setattr(self, f"transformer_{idx}", XMoEDecoderLayer(config))                                      #XMoE
            # setattr(self, f"transformer_{idx}", SkyDecoderLayer(idx, config))                                  #Skywork-MoE
            # setattr(self, f"transformer_{idx}", DeepSeekV3Transformer(idx, deepseekv3_config))                 #DeepSeek-V3
            # setattr(self, f"transformer_{idx}", DeepSeekV3Transformer(idx, deepseekv3_config, use_v3=False))   #DeepSeek-MoE
        self.norm = nn.LayerNorm(config.transformer_dim)

    def forward(self, input_ids, mask=None):
        X = self.embeddings(input_ids)
        cls_embed = self.cls_embed if not self.cls_last_layer else None

        if mask is None:
            mask = torch.ones_like(input_ids)

        if self.tied_weights:
            for idx in range(self.num_layers):
                if self.cls_last_layer and idx == self.num_layers - 1:
                    cls_embed = self.cls_embed
                X = self.transformer(X, mask, cls_embed)
                if cls_embed is not None:
                    # We always prepend the cls token into the first token
                    cls_embed = X[:, :1]
                    X = X[:, 1:]
        else:
            for idx in range(self.num_layers):
                if self.cls_last_layer and idx == self.num_layers - 1:
                    cls_embed = self.cls_embed
                X = getattr(self, f"transformer_{idx}")(X, mask, cls_embed)
                if cls_embed is not None:
                    # We always prepend the cls token into the first token
                    cls_embed = X[:, :1]
                    X = X[:, 1:]

        if cls_embed is not None:
            cls_embed = self.norm(cls_embed)
            return cls_embed
        else:
            X = self.norm(X) * mask[:, :, None]
            return X