"""
nanovllm/layers/linear.py - 并行线性层实现

本模块实现了支持张量并行的各种线性层变体。
张量并行是将模型权重切分到多个 GPU 上的技术，用于处理超大模型。

张量并行策略：
1. 列并行（Column Parallel）: 按输出维度切分
   - 输入: [B, input_size]
   - 权重: [output_size/tp, input_size]（每个 GPU）
   - 输出: [B, output_size/tp]（每个 GPU）
   - 无需通信

2. 行并行（Row Parallel）: 按输入维度切分
   - 输入: [B, input_size/tp]（每个 GPU）
   - 权重: [output_size, input_size/tp]（每个 GPU）
   - 输出: [B, output_size]（需要 all_reduce）

典型组合（MLP）:
    gate_up_proj: 列并行 -> intermediate_size/tp
    down_proj: 行并行 -> hidden_size（需 all_reduce）

典型组合（Attention）:
    qkv_proj: 列并行 -> (num_heads/tp) * head_dim
    o_proj: 行并行 -> hidden_size（需 all_reduce）
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    整除辅助函数
    
    确保可以整除，否则抛出异常。
    用于计算张量并行切分大小。
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类
    
    定义了权重加载接口和基本属性。
    
    Attributes:
        tp_dim (int): 张量并行切分的维度（0=列并行，1=行并行）
        tp_rank (int): 当前进程的 rank
        tp_size (int): 张量并行度
        weight (nn.Parameter): 权重参数
        bias (nn.Parameter): 偏置参数（可选）
    """

    def __init__(
        self,
        input_size: int,        # 输入维度
        output_size: int,       # 输出维度
        bias: bool = False,     # 是否有偏置
        tp_dim: int | None = None,  # 并行切分维度
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        
        # 偏置参数（可选）
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，由子类实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制型线性层（无并行）
    
    每个 GPU 都存储完整的权重。
    用于不需要并行的小型层（如某些归一化层中的投影）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制完整权重"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准线性变换"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层
    
    按输出维度切分权重，每个 GPU 计算部分输出。
    
    示意图（2路并行）：
    输入:   [B, input]
    权重:   [output/2, input] (GPU0)  [output/2, input] (GPU1)
    输出:   [B, output/2] (GPU0)      [B, output/2] (GPU1)
    
    无需通信，适合作为并行 MLP 的第一层。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输出维度按 tp_size 切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        从完整权重加载当前分片
        
        按 tp_dim=0（行维度）切分，取对应分片
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接计算，无需通信"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并的列并行线性层
    
    将多个列并行层合并为一个，用于 MLP 的 gate_proj 和 up_proj。
    
    示意图（gate + up，2路并行）：
    输入:   [B, hidden]
    权重:   [gate_size/2 + up_size/2, hidden] (GPU0)
    输出:   [B, gate_size/2 + up_size/2] (GPU0)
    
    合并的好处：
    - 减少 kernel 启动开销
    - 更好的内存访问模式
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # 各子层的输出大小列表
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        加载合并权重的特定分片
        
        Args:
            param: 目标参数
            loaded_weight: 单个子层的完整权重
            loaded_shard_id: 子层索引（0=gate, 1=up）
        """
        param_data = param.data
        # 计算在合并权重中的偏移
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 当前子层在当前 GPU 上的大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 定位到合并权重中的对应位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取当前 GPU 的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 并行线性层
    
    将 Q, K, V 三个投影合并为一个层，支持 GQA（K/V 头数可以少于 Q）。
    
    输出布局：[Q_heads * head_dim | K_heads * head_dim | V_heads * head_dim]
    
    每个 GPU 存储：
    - Q: num_heads/tp * head_dim
    - K: num_kv_heads/tp * head_dim
    - V: num_kv_heads/tp * head_dim
    
    权重加载时通过 shard_id ("q", "k", "v") 区分不同部分。
    """

    def __init__(
        self,
        hidden_size: int,           # 输入维度
        head_size: int,             # 每个头的维度
        total_num_heads: int,       # Q 的总头数
        total_num_kv_heads: int | None = None,  # KV 的总头数（GQA）
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        self.head_size = head_size
        # 当前 GPU 的头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        
        # 总输出大小
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        加载 Q/K/V 权重
        
        Args:
            param: 目标参数
            loaded_weight: 完整的 Q/K/V 权重
            loaded_shard_id: "q", "k", 或 "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # 计算在合并权重中的偏移和大小
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # 定位并复制
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层
    
    按输入维度切分权重，每个 GPU 计算部分结果后 all_reduce 汇聚。
    
    示意图（2路并行）：
    输入:   [B, input/2] (GPU0)      [B, input/2] (GPU1)
    权重:   [output, input/2] (GPU0)  [output, input/2] (GPU1)
    输出:   [B, output] (需要 all_reduce)
    
    配合列并行使用，在 MLP 的输出层和 Attention 的 O 投影使用。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度按 tp_size 切分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        从完整权重加载当前分片
        
        按 tp_dim=1（列维度）切分，取对应分片
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（带 all_reduce）
        
        每个 GPU 计算部分结果，然后 all_reduce 汇聚。
        偏置只在 rank 0 加（避免重复加）。
        """
        # 计算本地结果
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 多 GPU 时汇聚结果
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
