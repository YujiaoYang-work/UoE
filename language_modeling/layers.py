import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import nn
import math
# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
#     # Try to use FusedLayerNorm from Apex - this will trigger an error.
#     _ = LayerNorm(8, eps=1e-5)

from initialize import get_model_parallel_rank
from initialize import get_model_parallel_world_size
from fused_bias_gelu import bias_gelu_impl
import sys

#密切注意!!2.25修改版本目前只对keep_shape=False版本的代码的正确性进行了验证，
#为了正确无误的开源代码，还必须验证并跑通keep_shape=True的代码！！！！

# model_parallel_cuda_manual_seed(5)

def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    # with get_cuda_rng_tracker().fork():
    init_method(weight)

def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  params_dtype='torch.float32'
                                  ):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def get_model_world_size():
    return 4

def get_args():
    args = {
    "use_cpu_initialization": False,
    "params_dtype": torch.float32
    }
    return args

###############################################################
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""


    world_size = get_model_world_size()
    if (world_size == 1):
        return input_
    # stacked_input = torch.stack(input_)
    # output = torch.sum(stacked_input, dim=0, keepdim=False)
    output = None
    for i in input_:
        if output == None:
            output = i
        else:
            output = output + i
    # output = sum(input_)
    return output

def _reduce1(input_):
    """All-reduce the the input tensor across model parallel group."""


    world_size = get_model_world_size()
    if (world_size == 1):
        return input_
    # stacked_input = torch.stack(input_)
    # output = torch.sum(stacked_input, dim=0, keepdim=False)
    for i in range(len(input_)):
      input_[i] = torch.sum(input_[i], dim=0, keepdim=True)
      # print(input_[i].shape, "2211")

    output = input_.sum()

    return output

def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_model_world_size()
    # Bypass the function if we are using only 1 GPU.
    if (world_size == 1):
        return input_

    # Split along last dimension.
    output_list = split_tensor_along_last_dim(input_, world_size, contiguous_split_chunks=True)

    return output_list

def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_[0].dim() - 1

    output = torch.cat(input_, dim=last_dim).contiguous()

    return output

def copy_to_model_region(input_):
    return input_

def reduce_from_model_region(input_):
    return _reduce(input_)

def scatter_to_model_region(input_):
    return _split(input_)

def gather_from_model_region(input_):
    return _gather(input_)

def split_heads(X, num_head, head_dim):
    X = X.reshape(X.size(0), X.size(1), num_head, head_dim)
    X = X.transpose(1, 2)
    return X

def combine_heads(X, num_head, head_dim):
    X = X.transpose(1, 2)
    X = X.reshape(X.size(0), X.size(1), num_head * head_dim)
    return X

class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 skip_bias_add=False, use_cpu_initialization=False,
                 params_dtype=torch.float32):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size // world_size
        self.gather_output = gather_output
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.weight = nn.ParameterList()

        if use_cpu_initialization:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(self.output_size_per_partition,
                self.input_size, dtype=params_dtype))

                _initialize_affine_weight_cpu(weight_i, self.output_size,
                    self.input_size, self.output_size_per_partition,
                    0, init_method, stride=stride)
                self.weight.append(weight_i)
        else:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))

                _initialize_affine_weight_gpu(weight_i, init_method,
                                          partition_dim=0, stride=stride)
                self.weight.append(weight_i)

        if bias:
            self.bias = nn.ParameterList()
            if use_cpu_initialization:
                for i in range(world_size):
                    bias_i = Parameter(torch.empty(
                        self.output_size_per_partition, dtype=params_dtype))
                    bias_i.model_parallel = True
                    bias_i.bias.partition_dim = 0
                    bias_i.bias.stride = stride
                    with torch.no_grad():
                        bias_i.zero_()
                    self.bias.append(bias_i)
            else:
                for i in range(world_size):
                    bias_i = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
                    bias_i.model_parallel = True
                    bias_i.partition_dim = 0
                    bias_i.stride = stride
                    with torch.no_grad():
                        bias_i.zero_()
                    self.bias.append(bias_i)
        else:
            self.register_parameter('bias', None)
        self.world_size = world_size
        # print(self.weight, self.bias)
        # sys.exit(0)

    def forward(self, input_):
        input_ = copy_to_model_region(input_)
        output_ = []
        for i in range(self.world_size):
            bias = self.bias[i] if not self.skip_bias_add else None

            # output_.append(input.matmul(self.weight[i].t()) + bias)              #注意，纵向切割时，bias是在乘积的时候相加(gather之前)，横向切割时，是在乘积结束后相加(reduce之后)
            output_.append(F.linear(input_, self.weight[i], bias))              #这两句效果类似，或许就实验结果来看，上面这一种略微好一丢丢
        if self.gather_output:
            # All-gather across the partitions.
            output = _gather(output_)                    #dim按照实际情况进行更改
        else:
            output = output_
        if (self.skip_bias_add == True and self.bias):
            output_bias = torch.cat(tuple(self.bias), dim=-1)
        else:
            output_bias = None

        return output, output_bias

class ColumnParallelLinearWithMoE(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 skip_bias_add=False, use_cpu_initialization=False,
                 params_dtype=torch.float32):
        super(ColumnParallelLinearWithMoE, self).__init__()

        self.input_size = input_size
        self.output_size = output_size // world_size
        self.gather_output = gather_output
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.weight = nn.ParameterList()

        if use_cpu_initialization:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(self.output_size_per_partition,
                self.input_size, dtype=params_dtype))

                _initialize_affine_weight_cpu(weight_i, self.output_size,
                    self.input_size, self.output_size_per_partition,
                    0, init_method, stride=stride)
                self.weight.append(weight_i)
        else:
            for i in range(world_size):
                dtype = torch.float32
                weight_i = nn.Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=dtype))    #####, dtype=params_dtype

                _initialize_affine_weight_gpu(weight_i, init_method,
                                          partition_dim=0, stride=stride)
                self.weight.append(weight_i)

        if bias:
            self.bias = nn.ParameterList()
            if use_cpu_initialization:
                for i in range(world_size):
                    bias_i = Parameter(torch.empty(
                        self.output_size_per_partition, dtype=params_dtype))
                    bias_i.model_parallel = True
                    bias_i.bias.partition_dim = 0
                    bias_i.bias.stride = stride
                    with torch.no_grad():
                        bias_i.zero_()
                    self.bias.append(bias_i)
            else:
                for i in range(world_size):
                    bias_i = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
                    bias_i.model_parallel = True
                    bias_i.partition_dim = 0
                    bias_i.stride = stride
                    with torch.no_grad():
                        bias_i.zero_()
                    self.bias.append(bias_i)
        else:
            self.register_parameter('bias', None)
        self.world_size = world_size

    def forward(self, input_, idx_list, keep_shape=True, split_head=False):
        
        # if (len(input_.shape) == 3):
        #     bs, seq_len, d_model = input_.shape
        #     input_ = input_.reshape(bs, -1)

        # print(output_[0].shape, output.shape, len(output_), idx_list[0].shape, output_[0].shape)
        #带路由的gather操作：将output gather到货架上
        if (keep_shape):               #gather会增加额外的index操作和full/zero_操作，会大幅增加计算量（如图）
            bs, seq_len, d_model = input_.shape
            input_ = copy_to_model_region(input_)
            output = torch.zeros([bs, seq_len, self.output_size * self.world_size]).to(input_.device)
            for i in range(self.world_size):
                bias = self.bias[i] if not self.skip_bias_add else None
#            for i in range(self.world_size):      #注意：去除这一句可能带来问题
                if (len(idx_list[i]) != 0):
                    output[idx_list[i], :seq_len, i*self.output_size:(i+1)*self.output_size] = F.linear(input_[idx_list[i]], self.weight[i], bias)

            if (self.skip_bias_add == True and self.bias):
                output_bias = torch.cat(tuple(self.bias), dim=-1)
            else:
                output_bias = None

            return output, output_bias
        else:
            output_ = []
            for i in range(self.world_size):
                if (len(idx_list[i]) != 0):
                    x = input_[i]
                    x = copy_to_model_region(x)
                    bias = self.bias[i] if not self.skip_bias_add else None
                    # output_.append(input.matmul(self.weight[i].t()) + bias)              #注意，纵向切割时，bias是在乘积的时候相加(gather之前)，横向切割时，是在乘积结束后相加(reduce之后)

                    x = F.linear(x, self.weight[i], bias)
                    # x = torch.add(torch.matmul(x, self.weight[i].t()), bias)                                #未知问题：使用F.linear会导致结果精度变为float16,这是因为开启了混合精度训练

                    if split_head:
                        x = split_heads(x, num_head=self.world_size, head_dim=x.shape[-1]//self.world_size)
                    output_.append(x)              #这两句效果类似，或许就实验结果来看，上面这一种略微好一丢丢
                else:
                    output_.append([])

            return output_, self.bias

# ##################################################################################################################
# class ColumnParallelLinear(torch.nn.Module):
#     def __init__(self, input_size, output_size, world_size, bias=True, gather_output=True,
#                  init_method=init.xavier_normal_, stride=1,
#                  skip_bias_add=False, use_cpu_initialization=False,
#                  params_dtype=torch.float32):
#         super(ColumnParallelLinear, self).__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size // world_size
#         self.gather_output = gather_output
#         self.output_size_per_partition = divide(output_size, world_size)
#         self.skip_bias_add = skip_bias_add
#         self.weight = nn.ParameterList()
#
#         if use_cpu_initialization:
#             for i in range(world_size):
#                 weight_i = nn.Parameter(torch.empty(self.output_size_per_partition,
#                 self.input_size, dtype=params_dtype))
#
#                 _initialize_affine_weight_cpu(weight_i, self.output_size,
#                     self.input_size, self.output_size_per_partition,
#                     0, init_method, stride=stride)
#                 self.weight.append(weight_i)
#         else:
#             for i in range(world_size):
#                 weight_i = nn.Parameter(torch.empty(
#                 self.output_size_per_partition, self.input_size,
#                 device=torch.cuda.current_device(), dtype=params_dtype))
#
#                 _initialize_affine_weight_gpu(weight_i, init_method,
#                                           partition_dim=0, stride=stride)
#                 self.weight.append(weight_i)
#
#         if bias:
#             self.bias = nn.ParameterList()
#             if use_cpu_initialization:
#                 for i in range(world_size):
#                     bias_i = Parameter(torch.empty(
#                         self.output_size_per_partition, dtype=params_dtype))
#                     bias_i.model_parallel = True
#                     bias_i.bias.partition_dim = 0
#                     bias_i.bias.stride = stride
#                     with torch.no_grad():
#                         bias_i.zero_()
#                     self.bias.append(bias_i)
#             else:
#                 for i in range(world_size):
#                     bias_i = Parameter(torch.empty(
#                     self.output_size_per_partition,
#                     device=torch.cuda.current_device(),
#                     dtype=params_dtype))
#                     bias_i.model_parallel = True
#                     bias_i.partition_dim = 0
#                     bias_i.stride = stride
#                     with torch.no_grad():
#                         bias_i.zero_()
#                     self.bias.append(bias_i)
#         else:
#             self.register_parameter('bias', None)
#         self.world_size = world_size
#
#     def forward(self, input_):
#         input_ = copy_to_model_region(input_)
#         output_ = []
#         for i in range(self.world_size):
#             bias = self.bias[i] if not self.skip_bias_add else None
#
#             # output_.append(input.matmul(self.weight[i].t()) + bias)              #注意，纵向切割时，bias是在乘积的时候相加(gather之前)，横向切割时，是在乘积结束后相加(reduce之后)
#             output_.append(F.linear(input_[i], self.weight[i], bias))              #这两句效果类似，或许就实验结果来看，上面这一种略微好一丢丢
#         if self.gather_output:
#             # All-gather across the partitions.
#             output = _gather(output_)                    #dim按照实际情况进行更改
#         else:
#             output = output_
#         if (self.skip_bias_add == True and self.bias):
#             output_bias = torch.cat(tuple(self.bias), dim=-1)
#         else:
#             output_bias = None
#
#         return output, output_bias
# ####################################################################################################################

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, world_size, bias=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, use_cpu_initialization=False,
                 params_dtype=torch.float32):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        self.weight = nn.ParameterList()
        if use_cpu_initialization:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(self.output_size,
                                                    self.input_size_per_partition,
                                                    dtype=params_dtype))
                self.master_weight = _initialize_affine_weight_cpu(
                    weight_i, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)

                self.weight.append(weight_i)
        else:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(
                    self.output_size, self.input_size_per_partition,
                    device=torch.cuda.current_device(), dtype=params_dtype))

                _initialize_affine_weight_gpu(weight_i, init_method,
                                              partition_dim=1, stride=stride)

                self.weight.append(weight_i)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                    dtype=params_dtype))
                with torch.no_grad():
                    self.bias.zero_()
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.world_size = world_size

    def forward(self, input_):

        if not isinstance(input_, list):
            input_ = scatter_to_model_region(input_)
        output_ = []
        for i in range(len(input_)):
            # Matrix multiply.
            # output_.append(input_[i].matmul(self.weight[i].t()))
            output_.append(F.linear(input_[i], self.weight[i]))

        # All-reduce across all the partitions.
        # output = sum(output_)
        output = reduce_from_model_region(output_)
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output = output
            output_bias = self.bias

        return output, output_bias

class RowParallelLinearWithMoE(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, world_size, bias=True, keep_shape=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, use_cpu_initialization=False,
                 params_dtype=torch.float32):
        super(RowParallelLinearWithMoE, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        self.weight = nn.ParameterList()
        if use_cpu_initialization:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(self.output_size,
                                                    self.input_size_per_partition,
                                                    dtype=params_dtype))
                self.master_weight = _initialize_affine_weight_cpu(
                    weight_i, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)

                self.weight.append(weight_i)
        else:
            for i in range(world_size):
                weight_i = nn.Parameter(torch.empty(
                    self.output_size, self.input_size_per_partition,
                    device=torch.cuda.current_device(), dtype=params_dtype))

                _initialize_affine_weight_gpu(weight_i, init_method,
                                              partition_dim=1, stride=stride)

                self.weight.append(weight_i)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                    dtype=params_dtype))
                with torch.no_grad():
                    self.bias.zero_()
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.world_size = world_size

    def forward(self, input_, idx_list, keep_shape=True):
        
        #！！！这种linear函数涉及到维度的问题。对于transformer所需的三维输入，目前是按照Megatron官方代码直接进行处理
        #但是一般的linear是要先reshape成二维后再处理。到底采用哪种，后面还要斟酌一下

        if (keep_shape):
            shape1, shape2 = input_.shape[0], input_.shape[1]
            output = torch.zeros([shape1, shape2, self.output_size]).to(input_[0].device)
            if not isinstance(input_, list):
                input_ = scatter_to_model_region(input_)

            for i in range(len(input_)):
                if (len(idx_list[i]) != 0):
                    output[idx_list[i], :, :] = F.linear(input_[i][idx_list[i]], self.weight[i])
            if not self.skip_bias_add:
                output = output + self.bias if self.bias is not None else output
                output_bias = None
            else:
                output = output
                output_bias = self.bias
        else:
            output = []
            for i in range(self.world_size):
                if (len(idx_list[i]) != 0):

                    x = input_[i]
                    shape1, shape2 = x.shape[0], x.shape[1]
                    if not self.skip_bias_add:
                        output.append(F.linear(x, self.weight[i]) + self.bias if self.bias is not None else output)
                    else:
                        output.append(F.linear(x, self.weight[i]))
                else:
                    output.append([])
            output_bias = self.bias #!!!!!乱写的，后面要更正
        return output, output_bias

def RowParallelMatmul(input1, input2, norm_factor=1, all_reduce=True):
###注意：原版本中使用torch.baddbmm，不但有乘积还有乘以一个权重并加上一个偏差的操作，并且效率更高，后面应当写出这一版本
    shape1, shape2, shape3 = input1.shape[0], input1.shape[1], input2.shape[1]
    dtype = input1.dtype
    if not isinstance(input1, list):
        input1 = scatter_to_model_region(input1)
    if not isinstance(input2, list):
        input2 = scatter_to_model_region(input2)
    output_ = []
    for i in range(len(input1)):
        # Matrix multiply.
        matmul_result = torch.empty(
        shape1,
        shape2,
        shape3,
        dtype=dtype,
        device=torch.cuda.current_device())
        matmul_result = torch.baddbmm(matmul_result,
        input1[i],  # [b * np, s, hn]
        input2[i].transpose(1, 2),  # [b * np, hn, s]
        beta=0.0, alpha=(1.0 / norm_factor))
        output_.append(matmul_result)

        # output_.append(F.linear(input_[i], self.weight[i]))

    # All-reduce across all the partitions.
    # output = sum(output_)
    if all_reduce:
      output = reduce_from_model_region(output_)

    return output

def RowParallelMatmulWithMoE(input1, input2, world_size, idx_list, norm_factor=1, keep_shape=True, all_reduce=False):
###注意：原版本中使用torch.baddbmm，不但有乘积还有乘以一个权重并加上一个偏差的操作，并且效率更高，后面应当写出这一版本


    if (keep_shape):
        shape1, shape2, shape3 = input1.shape[0], input1.shape[1], input2.shape[1]
        dtype = input1.dtype
        if not isinstance(input1, list):
            input1 = scatter_to_model_region(input1)
        if not isinstance(input2, list):
            input2 = scatter_to_model_region(input2)

        output = torch.zeros([shape1, shape2, shape2]).to(input1[0].device)
        for i in range(len(input1)):
            if (len(idx_list[i]) != 0):
                matmul_result = torch.empty(
                len(idx_list[i]),
                shape2,
                shape3,
                dtype=dtype,
                device=torch.cuda.current_device())
                a = input1[i][idx_list[i], :]
                b = input2[i][idx_list[i], :]
                matmul_result = torch.baddbmm(matmul_result,
                a,  # [b * np, s, hn]
                b.transpose(-2, -1),  # [b * np, hn, s]                !!!!!!4.7日修改，暂未debug，可能出现问题
                beta=0.0, alpha=(1.0 / norm_factor))
                output[idx_list[i], :, :] = matmul_result
        return output
    else:
        output_ = []
        for i in range(world_size):
            if (len(idx_list[i]) != 0):        #为了加快速度，len(idx_list[i])可以直接通过参数传递获得

                # print(len(idx_list[i]), "sfjeofnr")
                # print(input1[i].shape, input2[i].shape, "inpit_shape222222222222222222222222")
                matmul_result  = torch.matmul(input1[i], input2[i].transpose(-2, -1))
                # a = input1[i]                                                         ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!原来的实现，由于不支持多头情况下的4为输出而弃用
                # b = input2[i]
                # shape1, shape2, shape3 = a.shape[0], a.shape[1], b.shape[1]
                # dtype = a.dtype
                # matmul_result = torch.empty(    #这里多了一个empty操作，可能增加运行时间。后面需要考察一下这种有empty的baddbmm操作核bmm操作究竟哪个更快，并选用更快的一个
                # shape1,               #之前这里是world_size，但是为什么是world_size呢？
                # shape2,
                # shape3,
                # dtype=dtype,
                # device=torch.cuda.current_device())
                # matmul_result = torch.baddbmm(matmul_result,
                # a,  # [b * np, s, hn]
                # b.transpose(1, 2),  # [b * np, hn, s]
                # beta=0.0, alpha=(1.0 / norm_factor))
                output_.append(matmul_result)
            else:
                output_.append([])

        if all_reduce:
            output_ = reduce_from_model_region(output_)
        return output_
    # All-reduce across all the partitions.
    # output = sum(output_)

def ColumnParallelMatmul(input1, input2, gather_output=True):
    input1 = copy_to_model_region(input1)
    if not isinstance(input1, list):
        input2 = scatter_to_model_region(input2.transpose(1, 2))

    output_ = []
    world_size = get_model_world_size()
    for i in range(world_size):
        output_.append(torch.bmm(input1, input2[i]))
    if gather_output:
        # All-gather across the partitions.
        output = _gather(output_)                    #dim按照实际情况进行更改
    else:
        output = output_

    return output

def ColumnParallelMatmulWithMoE(input1, input2, world_size, idx_list, keep_shape=True, combine_head=True):
    ###注意：这里的keep_shape同时关注输入和输出的格式，默认同一条运算路径上的keep_shape是一致的，如果不一致，会导致计算错误。后面为了体现兼容性，可以写出两种keep_shape,分别关注输入和输出的格式
    if (keep_shape):
        bs, seq_len, d_model = input2.shape
        
        input1 = copy_to_model_region(input1)
        if not isinstance(input1, list):
            input2 = scatter_to_model_region(input2)                                                        ###!!!!!!4.7日修改：去除.transpose(1, 2)，与V的乘积不用转置
        output = torch.zeros([bs, seq_len, d_model]).to(input1.device)
        for i in range(world_size):
            if (len(idx_list[i]) != 0):
                output[idx_list[i], :, i*d_model//world_size:(i+1)*d_model//world_size] = torch.bmm(input1[idx_list[i]], input2[i][idx_list[i]])
    else:
        #input: input1:list, len=4; input2:list, len=4, size=1->1/4
        output = []
        for i in range(world_size):                #！！！！！！有一个显著的问题：如果要多线程，就不能够各个线程公用一个变量，也就是说，
            if (len(idx_list[i]) != 0):            #所有的变量应该设为一个长度为4的变量数组！！！切记！！否则将会由于锁而阻碍进程
                matmul_result = torch.matmul(input1[i], input2[i])
                # a = input1[i]
                # b = input2[i]
                # print(a.shape, b.shape)
                # bs, seq_len, d_model = b.shape
                # #input2输入是一个完整的矩阵，下面要从这个完整的矩阵提取用于matmul的部分
                # if not isinstance(b, list):
                #     b = b[:, :, i*d_model//world_size:(i+1)*d_model//world_size]
                #不是，input2其实已经提取好了，直接相乘即可
                # matmul_result = torch.bmm(a, b)
                if combine_head:
                    matmul_result = combine_heads(matmul_result, num_head=world_size, head_dim=matmul_result.shape[-1])
                output.append(matmul_result)#.transpose(1, 2)?!!!!!!!
            else:
                output.append([])

    return output

class SparseMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, input_size, output_size, world_size, init_method=init.xavier_normal_, output_layer_init_method=init.xavier_normal_):
        super(SparseMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinearWithMoE(
            input_size,
            4 * input_size,
            world_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=False)

        self.bias_gelu_fusion = False#args.bias_gelu_fusion
        self.activation_func = F.gelu
        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu

        # Project back to h.

        self.dense_4h_to_h = RowParallelLinearWithMoE(
            4 * input_size,
            output_size,
            world_size,
            init_method=output_layer_init_method,
            skip_bias_add=False)

    def forward(self, hidden_states, idx_list, keep_shape=False):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states, idx_list, keep_shape)

        for i in range(len(intermediate_parallel)):
            if (len(intermediate_parallel[i]) != 0):
                if self.bias_gelu_fusion:
                    intermediate_parallel[i] = \
                            bias_gelu_impl(intermediate_parallel[i])
                else:
                    intermediate_parallel[i] = \
                        self.activation_func(intermediate_parallel[i])

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel, idx_list, keep_shape)
        return output, output_bias