"""
nanovllm/models/qwen3.py - Qwen3 模型实现

本模块实现了 Qwen3 系列模型的 Transformer 架构，是 Nano-vLLM 的模型层核心。
Qwen3 是一个 Decoder-only 的 Transformer 模型，采用以下技术：
- RoPE（旋转位置编码）
- RMSNorm（均方根归一化）
- SwiGLU 激活函数（MLP 使用 SiLU + 门控机制）
- GQA（分组查询注意力）- 支持 KV 头数少于 Q 头数

模型结构：
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding (词嵌入层)
│   ├── Qwen3DecoderLayer × N (Transformer 块)
│   │   ├── RMSNorm (input_layernorm)
│   │   ├── Qwen3Attention (自注意力)
│   │   ├── RMSNorm (post_attention_layernorm)
│   │   └── Qwen3MLP (前馈网络)
│   └── RMSNorm (最终归一化)
└── ParallelLMHead (语言模型头)

张量并行支持：
- QKV 投影使用列并行 (ColumnParallel)
- O 投影使用行并行 (RowParallel)
- MLP 的 gate_up 使用列并行，down 使用行并行
- Embedding 和 LMHead 使用词表并行
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力模块
    
    实现多头自注意力机制，支持：
    - GQA (Grouped Query Attention): KV 头数可以少于 Q 头数
    - RoPE (Rotary Position Embedding): 旋转位置编码
    - QK 归一化: 对 Q 和 K 进行 RMSNorm（可选）
    - Flash Attention: 高效的注意力计算
    - 张量并行: 在多 GPU 间分割注意力头
    
    计算流程:
    1. hidden_states -> QKV 投影 -> Q, K, V
    2. Q, K 可选归一化
    3. Q, K 应用 RoPE
    4. Flash Attention 计算
    5. O 投影 -> output
    """

    def __init__(
        self,
        hidden_size: int,          # 隐藏层维度
        num_heads: int,            # Query 头数
        num_kv_heads: int,         # Key/Value 头数 (GQA)
        max_position: int = 4096 * 32,  # 最大位置（RoPE 预计算范围）
        head_dim: int | None = None,    # 每个头的维度
        rms_norm_eps: float = 1e-06,    # RMSNorm epsilon
        qkv_bias: bool = False,         # QKV 投影是否有偏置
        rope_theta: float = 10000,      # RoPE 基础频率
        rope_scaling: dict | None = None,  # RoPE 缩放配置
    ) -> None:
        super().__init__()
        
        # ==================== 张量并行配置 ====================
        tp_size = dist.get_world_size()  # 张量并行度
        
        # 总头数（模型配置中的值）
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        # 当前进程的 Q 头数
        self.num_heads = self.total_num_heads // tp_size
        
        # KV 头数处理（GQA 支持）
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        # 当前进程的 KV 头数
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        # 每个头的维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # Q 和 KV 的总维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # ==================== 子模块定义 ====================
        # QKV 并行投影：将 hidden_states 投影到 Q, K, V
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # O 投影：将注意力输出投影回 hidden_size
        # 使用行并行，最后做 all_reduce
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # 某些 Qwen 配置会把 rope_theta 嵌在 rope_scaling 中；
        # 当前实现仍不支持真正的缩放 RoPE，只提取兼容的 theta。
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)

        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        
        # Flash Attention 模块
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # QK 归一化（Qwen3 的特性，没有 bias 时使用）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,      # 位置索引 [num_tokens]
        hidden_states: torch.Tensor,  # 输入 [num_tokens, hidden_size]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            positions: 每个 token 的位置索引
            hidden_states: 输入隐藏状态
        
        Returns:
            torch.Tensor: 注意力输出 [num_tokens, hidden_size]
        """
        # QKV 投影
        qkv = self.qkv_proj(hidden_states)
        # 分割 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 重塑为多头格式 [num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # QK 归一化（Qwen3 特有）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # Flash Attention 计算
        o = self.attn(q, k, v)
        
        # O 投影
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP（前馈网络）模块
    
    使用 SwiGLU 激活函数：
    MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    
    其中 gate_proj 和 up_proj 合并为一个投影 gate_up_proj，
    然后用 SiluAndMul 进行计算。
    
    张量并行：
    - gate_up_proj: 列并行（输出切分）
    - down_proj: 行并行（输入切分）+ all_reduce
    """

    def __init__(
        self,
        hidden_size: int,       # 输入/输出维度
        intermediate_size: int,  # 中间层维度
        hidden_act: str,         # 激活函数类型
    ) -> None:
        super().__init__()
        
        # 合并的 gate 和 up 投影
        # 输出维度是 [intermediate_size, intermediate_size]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # gate 和 up 各 intermediate_size
            bias=False,
        )
        
        # down 投影
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        
        # 确保使用 SiLU 激活
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        前向传播
        
        计算: down_proj(SiLU(gate) * up)
        其中 [gate, up] = gate_up_proj(x)
        """
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)  # SiLU(gate) * up
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Transformer 解码器层
    
    每层包含：
    1. 自注意力（带 Pre-Norm）
    2. MLP（带 Pre-Norm）
    3. 残差连接
    
    计算流程（Pre-Norm）：
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    
    实际实现中使用 fused add + norm 优化，
    减少内存访问次数。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        # 自注意力模块
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # MLP 模块
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 位置索引
            hidden_states: 输入或上一层的归一化输出
            residual: 残差连接的累积值
        
        Returns:
            (hidden_states, residual): 下一层的输入
        """
        # 第一个 LayerNorm（带 fused add）
        if residual is None:
            # 第一层：直接归一化
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：fused add + norm
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 自注意力
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 第二个 LayerNorm（带 fused add）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 主模型（不含 LM Head）
    
    组件：
    - embed_tokens: 词嵌入层（词表并行）
    - layers: N 层 Transformer 解码器
    - norm: 最终的 RMSNorm
    
    输出是最后一层的隐藏状态，需要通过 LM Head 转换为 logits。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        # 词嵌入层（词表并行）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,   # [num_tokens]
        positions: torch.Tensor,   # [num_tokens]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ids
            positions: 位置索引
        
        Returns:
            torch.Tensor: 最终隐藏状态 [num_tokens, hidden_size]
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 逐层计算
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终归一化（带 fused add）
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型
    
    在 Qwen3Model 基础上添加 LM Head，将隐藏状态转换为词表上的 logits。
    
    权重加载：
    - packed_modules_mapping: 定义权重名称映射
    - 原始权重 (q_proj, k_proj, v_proj) 被合并为 qkv_proj
    - 原始权重 (gate_proj, up_proj) 被合并为 gate_up_proj
    
    权重共享：
    - 如果 tie_word_embeddings=True，LM Head 与 embed_tokens 共享权重
    """
    
    # 权重名称映射：原始名称 -> (合并后名称, 分片ID)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        
        # 主模型
        self.model = Qwen3Model(config)
        
        # 语言模型头（词表并行）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播（只计算隐藏状态）
        
        Returns:
            torch.Tensor: 隐藏状态 [num_tokens, hidden_size]
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 logits
        
        将隐藏状态通过 LM Head 转换为词表上的 logits。
        在 prefill 阶段，只取每个序列最后一个 token 的 logits。
        
        Args:
            hidden_states: 隐藏状态
        
        Returns:
            torch.Tensor: logits [num_seqs, vocab_size]
        """
        return self.lm_head(hidden_states)
