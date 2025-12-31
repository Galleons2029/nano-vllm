# Nano-vLLM 项目讲解

这是一份面向阅读代码的快速说明，帮助你理解 Nano-vLLM 的整体设计、核心模块与使用方式。

## 项目概览
- 目标：用约 1.2k 行 Python 代码复刻 vLLM 核心能力，提供高效的离线推理。
- 模型：当前内置 `Qwen3-0.6B`（`nanovllm/models/qwen3.py`），接口与 vLLM 接近。
- 依赖：Python 3.10–3.12，PyTorch 2.4+、Triton 3.0+、Transformers 4.51+、Flash-Attn、xxhash（见 `pyproject.toml`）。
- 硬件：默认使用 GPU，支持多卡张量并行（`tensor_parallel_size`）。

## 目录速览
- `nanovllm/llm.py`：对外暴露的 `LLM` 类，继承引擎实现。
- `nanovllm/engine/`：调度与执行核心  
  - `llm_engine.py`：进程初始化、请求流转、生成循环。  
  - `scheduler.py`：prefill/decode 调度、KV 块预占与抢占。  
  - `block_manager.py`：KV 缓存块管理与前缀复用（哈希 + 引用计数）。  
  - `model_runner.py`：模型加载、KV 分配、Flash-Attn 前向、采样、CUDA Graph。  
  - `sequence.py`：请求状态机与 token/block 元数据。
- `nanovllm/layers/`：自实现/并行化的基础算子（并行 Linear、RMSNorm、RoPE、Flash-Attn 调用、采样器）。
- `nanovllm/utils/`：上下文管理与权重加载（支持 packed 权重拆分）。
- 示例与工具：`example.py`（快速使用）、`bench.py`（吞吐基准）。

## 运行流程（高层）
1) 初始化 `LLM`  
   - 生成 `Config`（`nanovllm/config.py`），读取 HF 配置与 tokenizer，确定 EOS、最大长度等。  
   - 按 `tensor_parallel_size` 启动多个 `ModelRunner` 进程（主进程 rank=0）。  
   - 绑定 `Scheduler` 和 `BlockManager`，注册退出清理。
2) 提交请求  
   - `add_request` 将字符串或 token 序列封装为 `Sequence`（带温度、最大生成长度、KV 块表等）放入等待队列。
3) 调度与批处理  
   - `schedule` 先进行 prefill：在 `max_num_batched_tokens` / `max_num_seqs` 约束下批量分配 KV 块；若资源不足会中断批次。  
   - 当无新的 prefill 时进入 decode：对运行中的序列逐个补写 KV 块；若 KV 满则抢占尾部序列（preempt）腾出空间。
4) 模型执行 (`model_runner.py`)  
   - Prefill：构造按块的 `block_tables` 与 `slot_mapping`，利用 Flash-Attn 可变长度接口 + 前缀缓存。  
   - Decode：仅追加最后一个 token，直接写入 KV cache 并用 Flash-Attn with KV-cache。  
   - 多卡时 rank0 采样，其他 rank 只前向；采样用自实现的温度缩放 + Gumbel 最大似然近似（`layers/sampler.py`）。  
   - 关闭 `enforce_eager` 时，对典型 batch size 捕获 CUDA Graph，decode 阶段复用以降低 launch 开销。
5) 结果返回  
   - `Scheduler.postprocess` 追加 token，命中 EOS 或达到 `max_tokens` 时回收 KV 块并标记完成。  
   - `LLM.generate` 持续循环直至所有序列完成，按提交顺序解码文本返回。

## 核心设计与优化
- **前缀缓存 + 块复用**：`block_manager.py` 使用 xxhash + 引用计数维护 KV 块；相同前缀仅存一份，避免重复计算与显存浪费。  
- **块化 KV 管理**：默认块大小 256 token，可按需扩展；允许抢占与回收，支持高并发下的动态批处理。  
- **张量并行**：自定义列并行/行并行 Linear、词表并行 Embedding/LM Head；权重加载时按 shard 自动切片（`layers/linear.py`, `layers/embed_head.py`）。  
- **高效注意力**：前向调用 Flash-Attn（prefill 用 varlen，decode 用 kvcache），并用 Triton 内核 `store_kvcache` 写 KV。  
- **CUDA Graph**：对 decode 常见 batch size 预捕获图（`capture_cudagraph`），减少内核 launch 开销。  
- **编译加速**：RMSNorm、SiluAndMul、Sampler 等小算子均用 `torch.compile`，减少 Python 开销。  
- **显存自适应**：`allocate_kv_cache` 按 `gpu_memory_utilization`、峰值/当前显存，计算可用 KV 块数并分配。

## 使用方法
### 安装
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```
需要提前下载模型权重（示例使用 `Qwen3-0.6B`）：
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 最小示例（见 `example.py`）
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling)
print(outputs[0]["text"])
```

### 简单基准（见 `bench.py`）
```bash
python bench.py
```
默认生成 256 条随机序列，打印总 token 数、耗时与吞吐。

## 主要配置项（`nanovllm/config.py`）
- `max_num_batched_tokens`：单批最大 token 数（含 prompt + decode，影响显存/吞吐）。  
- `max_num_seqs`：单批最大序列数。  
- `max_model_len`：模型上下文长度上限（会与 HF 配置取最小）。  
- `tensor_parallel_size`：张量并行分片数。  
- `enforce_eager`：为 True 时禁用 CUDA Graph。  
- `gpu_memory_utilization`：可用显存比例，用于计算 KV 块数量。  
- `kvcache_block_size` / `num_kvcache_blocks`：KV 块大小与总块数（默认自动估算）。

## 模型实现要点
- 结构：标准 Decoder-only Transformer，RoPE 位置编码，RMSNorm，SwiGLU (`SiluAndMul`) MLP。  
- 注意力：支持独立 KV 头数；Q/K/V 由并行化 QKV 投影得到；输出经行并行 Linear 规约。  
- 权重加载：支持 packed 权重映射（`packed_modules_mapping`），从 `.safetensors` 按 shard 写入各并行分片。

## 可继续探索的方向
- 增加其他 HF 模型适配（补充新的 `*_ForCausalLM`）。  
- 引入 paged KV cache 与更细粒度的预占/抢占策略。  
- 集成更多采样策略（top-k/p、penalty 等）。  
- 在多卡下扩展 pipeline 并行或张量并行 + ZeRO 权重切分。

以上内容概括了项目的关键结构与运行路径，可结合源码快速定位实现细节。
