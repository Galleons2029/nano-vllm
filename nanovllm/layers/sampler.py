"""
nanovllm/layers/sampler.py - Token 采样器

本模块实现从 logits 到 token ID 的采样过程。
采样是语言模型生成的核心步骤，决定了输出的多样性和质量。

采样方法：
本实现使用 Gumbel-max 采样，等价于多项式（categorical）采样，
但更适合 GPU 并行计算。

Gumbel-max 原理：
- 给每个 logit 加上 Gumbel(0,1) 噪声
- 然后取 argmax
- 数学上等价于按 softmax 概率采样

具体来说：
1. 对于概率 p_i = softmax(logit_i / T)
2. Gumbel 噪声 g_i = -log(-log(U_i))，U_i ~ Uniform(0,1)
3. argmax_i(logit_i / T + g_i) 与从 p 采样等价

实现技巧：
- 使用 exponential 分布：exp(1) = -log(U) when U ~ Uniform(0,1)
- 除法代替加法：p_i / exp(1) 的 argmax 等价于 Gumbel-max
- 更数值稳定且高效

温度参数：
- temperature = 1.0：标准采样
- temperature < 1.0：更确定性（偏向高概率）
- temperature > 1.0：更随机（增加多样性）
- temperature → 0：退化为 greedy（取 argmax）
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器
    
    使用 Gumbel-max 技巧实现高效的 GPU 采样。
    支持温度调节。
    
    为什么用 Gumbel-max 而不是 torch.multinomial：
    1. torch.multinomial 在 GPU 上相对较慢
    2. Gumbel-max 完全向量化，无同步开销
    3. 可以与 torch.compile 良好结合
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        从 logits 采样 token
        
        Args:
            logits: 模型输出的原始分数 [batch_size, vocab_size]
            temperatures: 每个序列的温度 [batch_size]
        
        Returns:
            torch.Tensor: 采样的 token ID [batch_size]
        
        计算步骤：
        1. 温度缩放：logits / temperature
        2. Softmax 得到概率
        3. Gumbel-max 采样：
           - 生成指数分布噪声
           - probs / exp_noise 的 argmax
        
        数学等价性：
        probs / exp_noise 的 argmax 等价于
        log(probs) - log(exp_noise) 的 argmax =
        log(probs) + log(U) 的 argmax（Gumbel-max 变体）
        """
        # 1. 温度缩放
        #    转换为 float 避免精度损失
        #    就地操作节省内存
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 2. Softmax 得到概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 3. Gumbel-max 采样
        #    - exponential_(1) 生成 Exp(1) 噪声
        #    - clamp_min 防止除零
        #    - 除法后取 argmax
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        return sample_tokens

