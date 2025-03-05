"""
This file is from https://github.com/mlpen/Nystromformer
"""

import torch
import torch.nn as nn
import numpy as np
import math
from attention722_2 import Attention ###!!!!!!!!!!!!!!!!注意后面要改回来！！！！！！！！！！
# from attention import SparseSelfAttention
# from sparse_attention2D1025 import SparseSelfAttention

from sparse_attention2D_fine3 import SparseSelfAttention
# from sparse_attention_v3 import SparseSelfAttention

# from mlps823 import U_MLP    #低速高精度版2D
# from mlps25 import U_MLP
# from mlps1D import U_MLP
from mlps8233 import U_MLP    #低速高精度版2D
# from mlps2 import U_MLP    #低速高精度版2D
# from sparse_mlp_v3 import U_MLP


from attention_transformer_ls import AttentionLS

from sparse_attention_module import ModuleFormerAttention
from switch_transformer import SwitchTransformerBlock
from transformers.activations import get_activation
from typing import Optional, Tuple, Union
from moe import MoE
# torch.autograd.set_detect_anomaly(True)
from modeling_1127 import MLADecoderLayer

from DeepSeekV3 import Transformer as DeepSeekV3Transformer
from DeepSeekV3 import ModelArgs as DeepSeekV3ModelArgs
from process_v3 import precompute_freqs_cis

import sys

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
        # if config.attn_type == 'lsta':
        #     self.mha = AttentionLS(config)
        # else:
        #     self.mha = Attention(config)
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

class ParallelTransformer(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.transformer_dim)
        self.mha = SparseSelfAttention(config)
        # self.mha = Attention(config)
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

        # self.mlpblock = MH_MLP(config)
        # self.mlpblock = S_MH_MLP(config)
        self.mlpblock = U_MLP(config)
        # self.mlpblock = MH_U_MLP(config)

    def forward(self, X, mask, cls_embed=None):
        # numpy_array = mask.cpu().numpy()
        # import numpy as np
        # np.savetxt('tensor3.txt', numpy_array, fmt='%d')
        # sys.exit(0)

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
        # sys.exit(0)
        return X

class ModuleFormerBlock(nn.Module):
    def __init__(self, config):
        """
        Initialize the ModuleFormerBlock module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = ModuleFormerAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = MoE(
                input_size=config.n_embd,
                head_size=config.ffd_hidden,
                num_experts=config.n_mlp_experts,
                top_k=config.k_mlp,
                bias=False,
                activation=get_activation(config.activation_function),
                acc_aux_loss=False,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
                gate_type=config.gate_type,
            )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def get_aux_loss_and_clear(self):
        """
        Get auxiliary loss and clear auxiliary loss accumulators in the attention and MLP layers.

        Returns:
            torch.Tensor: Auxiliary loss.
        """

        return self.attn.q_proj.get_aux_loss_and_clear() + self.mlpf.get_aux_loss_and_clear()

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """
        Forward pass of the ModuleFormerBlock module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            head_mask (Optional[torch.FloatTensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
            Tuple containing outputs or optional attention weights.
        """
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        hidden = attn_outputs[1]
        att_aux_loss = attn_outputs[2]

        hidden_states = hidden_states + self.resid_dropout(attn_output)
        x_mlp, mlp_aux_loss = self.mlpf(self.ln_2(hidden_states))
        hidden_states = hidden_states + self.resid_dropout(x_mlp)

        # aux_loss = att_aux_loss + mlp_aux_loss
        return hidden_states#, hidden, aux_loss) + attn_outputs[3:]

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

        if self.tied_weights:
            # self.transformer = Transformer(config)
            # self.transformer = ParallelTransformer(config)
            self.transformer = MLADecoderLayer(config)
        else:
            for idx in range(self.num_layers):
                # setattr(self, f"transformer_{idx}", Transformer(config))
                setattr(self, f"transformer_{idx}", ParallelTransformer(config, deepseekv3_config))
                # setattr(self, f"transformer_{idx}", DeepSeekV3Transformer(idx, deepseekv3_config))
                # setattr(self, f"transformer_{idx}", DeepSeekV3Transformer(idx, config, deepseekv3_config))
                # setattr(self, f"transformer_{idx}", MLADecoderLayer(config))
                # setattr(self, f"transformer_{idx}", SwitchTransformerBlock(config))
                # setattr(self, f"transformer_{idx}", ModuleFormerBlock(config))

        self.norm = nn.LayerNorm(config.transformer_dim)

    def forward(self, input_ids, mask=None):
        X = self.embeddings(input_ids)
        cls_embed = self.cls_embed if not self.cls_last_layer else None

        if mask is None:
            mask = torch.ones_like(input_ids)
        # print(mask.shape, "Dfvmdofjivdjfoivdsikfs")
        # count_non_one = (mask == 0).sum().item()
        # print(count_non_one, "dfvmjodijfivjdifv")
        # long_array = mask.cpu().numpy()
        #
        # # 保存到 txt 文件
        # np.savetxt('long_tensor.txt', long_array, fmt='%d')
        # import sys
        # sys.exit(0)
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