"""
nanovllm/sampling_params.py - 采样参数配置类

本模块定义了控制文本生成行为的采样参数。
采样参数决定了模型如何从概率分布中选择下一个 token。

采样过程简述：
1. 模型输出 logits（未归一化的对数概率）
2. 应用温度缩放: logits = logits / temperature
3. 转换为概率分布: probs = softmax(logits)
4. 从概率分布中采样选择 token
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    采样参数配置类
    
    Attributes:
        temperature (float): 温度系数，用于控制生成的随机性
            - 较高的温度（如 1.0-2.0）使分布更平坦，输出更多样/随机
            - 较低的温度（如 0.1-0.5）使分布更尖锐，输出更确定/保守
            - 必须 > 0（不支持贪婪解码 temperature=0）
            
        max_tokens (int): 最大生成 token 数
            - 达到此数量后停止生成
            - 默认值 64
            
        ignore_eos (bool): 是否忽略结束符 (End of Sequence)
            - True: 即使遇到 EOS token 也继续生成，直到达到 max_tokens
            - False: 遇到 EOS token 立即停止生成
            - 主要用于基准测试，确保生成固定长度的输出
    
    注意：
        当前实现使用 Gumbel-Max 采样技巧，这是一种高效的采样方法，
        等价于从 softmax 分布中采样，但避免了显式的累积分布计算。
    """
    temperature: float = 1.0    # 温度系数，控制随机性
    max_tokens: int = 64        # 最大生成 token 数
    ignore_eos: bool = False    # 是否忽略结束符

    def __post_init__(self):
        """
        参数验证
        
        temperature 必须大于一个很小的正数，因为：
        1. temperature=0 对应贪婪解码，需要特殊处理（当前未实现）
        2. 太小的温度会导致数值不稳定
        """
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
