# Mini Transformer (PyTorch)

一个用于动手学习 Transformer / GPT 基础原理的最小项目。

## 深入学习资料

- [完整源码学习手册](CODE_WALKTHROUGH.md)：从 tokenizer、训练数据、Attention 公式到 Prefill / KV Cache / Decode 的逐步讲解
- [完整模型流程图](model_prefill_flowchart.svg)
- [模型主干与训练放大图](model_training_detail.svg)
- [Attention 与 KV Cache 放大图](attention_kv_cache_detail.svg)

## 文件

- `data.txt`：简单英文语料库
- `tokenizer.py`：简单 word-level tokenizer
- `model.py`：手写 Multi-Head Self-Attention、FFN、MoE、Transformer Block、Language Model
- `train.py`：Next Token Prediction 训练代码
- `generate.py`：加载训练结果并生成文本
- `test_prefill.py`：prefill、KV cache 和滑窗生成测试
- `test_moe.py`：专家路由、训练梯度、缓存一致性和 checkpoint 测试
- `requirements.txt`：依赖

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

训练完成后会生成：

```text
model.pt
```

模型固定使用 MoE：每层 4 个 FFN 专家，每个 token 选择其中 2 个；推理使用 KV cache。
建议学习时先调小 `train_steps`。checkpoint 同时保存模型配置，`generate.py` 按配置加载。
结构变更前保存的 checkpoint 需要用当前 `train.py` 重新训练生成。

## MoE 怎样工作

每个 Block 由 Attention 和 MoE（Mixture of Experts）组成，KV cache 仅在推理路径中维护。
每个专家都是现有的 `FeedForward`，拥有各自独立的参数。

```text
token 向量 [C]
  → router 打分并 softmax，得到 4 个专家的概率
  → 选择概率最大的 2 个专家，并重新归一化这两个权重
  → 把 token 送入选中的 FFN 专家
  → 两个专家输出按权重相加，得到 [C]
```

例如概率为 `[0.1, 0.6, 0.2, 0.1]`，选择专家 1、2，权重变成 `0.75、0.25`：

```text
MoE(x) = 0.75 * expert_1(x) + 0.25 * expert_2(x)
```

训练损失为 `交叉熵 + 0.01 × 各层平均负载均衡损失`，鼓励 router 分散使用专家。
默认 top-2；若改为 top-1，归一化后的权重恒为 1，router 依靠辅助损失学习。
这里的 top-k 选择的是专家，与生成时选择候选 token 的 Top-K 采样不同。

建议阅读顺序：`FeedForward → MoE.forward → TransformerBlock → TransformerLanguageModel.forward`。
详细 shape、辅助损失公式和参数量见[学习手册的 MoE 小节](CODE_WALKTHROUGH.md#moe用多个专家替换-ffn)。
实现使用直接的专家循环，便于阅读；没有容量限制、token 丢弃或分布式通信，也不保证比普通 FFN 更快。

## 生成文本

```bash
python generate.py
```

生成分为两个阶段：

1. `model.prefill(input_ids)` 一次处理当前上下文窗口，并为每一层建立 K/V cache。
2. cache 未满时，`model.decode(next_token, kv_cache)` 只处理一个新 token 并更新 cache。
3. Continuous batching 改进的是多请求调度；把它和 prefill、decode、KV cache 串起来看，就能理解它如何减少请求等待和 GPU 空闲。
4.  Temperature（温度）：控制概率分布的平滑程度，影响生成随机性。
    Top-K：只允许从概率最高的 K 个 Token 中采样。
    Top-P（Nucleus Sampling）：只允许从累计概率达到 P 的最小 Token 集合中采样。
    Repetition Penalty（重复惩罚）：降低已经出现过的 Token 再次出现的倾向
    multinomial:随机采样
5.  llama.cpp / onnx /  vLLM / SGLang / TensorRT-LLM / Triton Inference Server
示例：

```python
logits, kv_cache = model.prefill(input_ids)
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
# 返回的 logits 用于预测 next_token 后面的 token。
logits, kv_cache = model.decode(next_token, kv_cache)
```

模型使用 learned absolute position embedding。cache 达到 `max_seq_len` 后，
`generate.py` 会用最近一个窗口重新执行 prefill，而不是继续 decode，以保持原有滑窗生成语义。
prompt 超过 `max_seq_len` 时，也只会 prefill 最近一个窗口。当前模型没有 padding mask，
批量 prefill 仅适用于长度相同且不含 padding 的输入。

训练的 `model(x, y)` 走 `_forward_impl`，普通 Attention 只返回输出，不创建或收集 KV cache。
`prefill/decode` 走 `_forward_with_cache`，通过各层的 `forward_with_cache` 建立和更新缓存，无需启用开关。
两条路径都计算注意力所需的 Q/K/V，区别在于是否将 K/V 留给后续推理复用。

运行测试：

```bash
python -m unittest -v test_prefill.py test_moe.py
```

建议学习时先把 `train_steps` 改成较小值，并在 `model.py` 中打印 Q、K、V、scores、attention_weights 的 shape。
