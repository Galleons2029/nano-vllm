"""
nanovllm/utils/loader.py - 模型权重加载器

本模块负责从 SafeTensors 格式加载预训练权重到模型。
特别支持 packed modules（合并的权重矩阵）的处理。

背景知识：
1. SafeTensors 格式：HuggingFace 推出的安全张量存储格式
   - 比 pickle 更安全（无代码执行风险）
   - 支持内存映射和懒加载

2. Packed Modules（合并权重）：
   - 某些实现会合并多个矩阵以优化计算
   - 例如：Q、K、V 权重合并为 QKV 权重
   - 例如：gate_proj 和 up_proj 合并为 gate_up_proj
   - 预训练权重是分开的，需要正确加载到合并矩阵的对应位置

权重加载流程：
1. 遍历 safetensors 文件中的所有权重
2. 检查是否在 packed_modules_mapping 中
   - 是：转换名称，使用 shard_id 指示加载位置
   - 否：直接加载
3. 调用参数的 weight_loader 方法执行实际加载

示例：
原始权重: model.layers.0.self_attn.q_proj.weight
模型参数: model.layers.0.self_attn.qkv_proj.weight (Q,K,V 合并)
packed_modules_mapping: {"q_proj": ("qkv_proj", 0)}
加载时: 将 q_proj 权重加载到 qkv_proj 的第 0 个分片
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载器
    
    简单地将加载的权重复制到参数中。
    用于非合并权重的直接加载。
    
    Args:
        param: 目标参数
        loaded_weight: 加载的权重张量
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重
    
    从指定路径的 safetensors 文件加载权重到模型。
    支持 packed modules（合并权重）的正确处理。
    
    Args:
        model: 要加载权重的模型
        path: 权重文件所在目录
    
    工作流程：
    1. 获取模型的 packed_modules_mapping（如果有）
    2. 遍历目录下所有 .safetensors 文件
    3. 对每个权重：
       a. 检查是否属于 packed module
       b. 是：转换参数名，使用 weight_loader 加载到对应分片
       c. 否：直接使用 weight_loader 或 default_weight_loader 加载
    
    Packed Modules 机制：
    - 模型定义 packed_modules_mapping 字典
    - 键：原始权重名中的组件（如 "q_proj"）
    - 值：(合并后的名称, 分片索引)（如 ("qkv_proj", 0)）
    - 加载时将权重放入合并矩阵的正确位置
    
    Weight Loader：
    - 每个参数可以有自定义的 weight_loader 属性
    - 用于处理张量并行、分片等特殊情况
    - 没有时使用 default_weight_loader
    """
    # 获取 packed modules 映射（模型可能没有定义）
    # 例如 Qwen3 定义了 QKV 合并和 gate_up 合并
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 遍历所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # 使用内存映射打开文件（高效处理大文件）
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重
            for weight_name in f.keys():
                # 检查是否属于 packed module
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 属于 packed module，需要特殊处理
                        v, shard_id = packed_modules_mapping[k]
                        # 转换参数名（如 q_proj -> qkv_proj）
                        param_name = weight_name.replace(k, v)
                        # 获取目标参数
                        param = model.get_parameter(param_name)
                        # packed module 必须有自定义 weight_loader
                        weight_loader = getattr(param, "weight_loader")
                        # 加载权重到指定分片
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 不属于 packed module，直接加载
                    param = model.get_parameter(weight_name)
                    # 尝试获取自定义 weight_loader，否则使用默认
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 加载权重
                    weight_loader(param, f.get_tensor(weight_name))

