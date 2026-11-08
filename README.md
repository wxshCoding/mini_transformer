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
- `model.py`：手写 Multi-Head Self-Attention、FFN、Transformer Block、Language Model
- `train.py`：Next Token Prediction 训练代码
- `generate.py`：加载训练结果并生成文本
- `test_prefill.py`：prefill、KV cache 和滑窗生成测试
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

## 生成文本

```bash
python generate.py
```

生成分为两个阶段：

1. `model.prefill(input_ids)` 一次处理当前上下文窗口，并为每一层建立 K/V cache。
2. cache 未满时，`model.decode(next_token, kv_cache)` 只处理一个新 token 并更新 cache。

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

运行 prefill 测试：

```bash
python -m unittest test_prefill.py
```

建议学习时先把 `train_steps` 改成较小值，并在 `model.py` 中打印 Q、K、V、scores、attention_weights 的 shape。
