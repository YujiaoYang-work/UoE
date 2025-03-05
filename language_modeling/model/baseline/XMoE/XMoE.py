from typing import Callable, Dict, Tuple, Optional
from deepspeed import comm as dist

import torch
from torch import Tensor
import torch.nn.functional as F

import copy
import math


class Experts(torch.nn.Module):
	def __init__(self, expert, num_local_experts=1):
		super(Experts, self).__init__()

		self.yyh_local_experts = torch.nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])

	def forward(self, inputs, inputs_weight, top_idx):
		# inputs: (s, m), inputs_weight: (s, e)
		expert_output = torch.zeros_like(inputs)
		out_non_zero_ratio = None
		for e_idx, expert in enumerate(self.yyh_local_experts):
			token_idx = top_idx[:, e_idx]  # (capacity)
			these_tokens = inputs[token_idx]  # (capacity, dim)

			out = expert(these_tokens)

			if type(out) is tuple:
				if out_non_zero_ratio is None:
					out_non_zero_ratio = out[2]
				else:
					out_non_zero_ratio += out[2]

				out = out[0]  # Ignore the bias term for now

			expert_output[token_idx] += out * inputs_weight[:, e_idx][token_idx].unsqueeze(-1).type_as(inputs)

		return expert_output, out_non_zero_ratio / len(self.yyh_local_experts)

class SparseMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False,
                 threshold: float = -1.0,
                 placeholder_expert: bool = False,
                 view_num: int = 1,
                 scale_moe: bool = False):
        # most of arg is unused since this is a simplified MoE
        super(SparseMLP, self).__init__()

        args = get_args()
        self.use_base_layer = args.use_base_layer
        self.use_topk = args.use_topk
        self.use_threshold = args.use_threshold
        self.use_hash_layer = args.use_hash_layer
        if self.use_base_layer + self.use_topk + self.use_threshold + self.use_hash_layer != 1:
            raise Exception("只能指定有且只有一个路由！")

        self.num_experts = num_experts
        self.num_local_experts = num_experts

        self.experts = Experts(expert, num_experts)
        if self.use_hash_layer:
            self.gate = HashRouter(num_experts, args.padded_vocab_size, capacity_factor, eval_capacity_factor, min_capacity)
        else:
            self.gate = TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                   min_capacity, noisy_gate_policy, drop_tokens, use_rts, threshold=threshold,
                                   placeholder_expert=placeholder_expert, view_num=view_num,
                                   num_local_experts=self.num_local_experts, scale_moe=scale_moe)

        self.gating_function = main_thresholdGating

    def forward(self, hidden_states, now_training_process, enc_input_ids=None):
        sequence_len, bsz_size, d_model = hidden_states.shape

        reshaped_input = hidden_states.reshape(-1, d_model)

        if self.use_hash_layer:
            enc_input_ids = enc_input_ids.t().reshape(-1) # (bsz, seq_len) -> (seq_len, bsz) -> (seq_len * bsz,)
            combine_weights, gate_info, top_idx = self.gate(enc_input_ids)

            exp_counts = None
            l_aux = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            l_aux, combine_weights, _, exp_counts, gate_info, top_idx = self.gate(reshaped_input,
                                                                                        None,
                                                                                        in_logits=None,
                                                                                        now_training_process=now_training_process,
                                                                                        gating_function=None,
                                                                                        use_base_layer=self.use_base_layer,
                                                                                        use_topk=self.use_topk,
                                                                                        use_threshold=self.use_threshold)

        expert_output, non_zero_ratio = self.experts(reshaped_input, combine_weights, top_idx)

        gate_info["non_zero_ratio"] = non_zero_ratio

        output = expert_output.reshape(hidden_states.shape)

        return output, l_aux, exp_counts, gate_info


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, moe=False, enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        in_hidden_size = args.hidden_size
        ffn_hidden_size = args.ffn_hidden_size

        self.scale_moe = moe and args.scale_moe
        if moe:
            in_hidden_size = args.hidden_size // args.num_ffn_heads
            ffn_hidden_size = args.moe_ffn_hidden_size

        if self.scale_moe:
            self.dense_small_to_h = torch.nn.Linear(in_hidden_size // args.topk, in_hidden_size)
            self.dense_h_to_small = torch.nn.Linear(in_hidden_size, in_hidden_size // args.topk)

        self.add_bias = config.add_bias_linear

        if config.gated_linear_unit:
            ffn_hidden_size *= 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            in_hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.yyh_relu:
            def vanilla_relu(x):
                return F.relu(x)
            self.activation_func = vanilla_relu
        elif args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            ffn_hidden_size,
            in_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

    def get_key_weight(self):
        return self.dense_h_to_4h.weight.mean(dim=0, keepdim=True)  # (1, input_dim)

    def forward(self, hidden_states):

        if self.scale_moe:
            hidden_states = self.dense_small_to_h(hidden_states)

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
            assert hasattr(self, "__flops__") or self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        non_zero_ratio = (intermediate_parallel.view(-1) > 0).float().mean(dim=-1)
        # intermediate_parallel = F.softmax(intermediate_parallel, dim=-1)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)

        if self.scale_moe:
            output = self.dense_h_to_small(output)

        return output, output_bias, non_zero_ratio
