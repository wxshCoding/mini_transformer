# Mini Transformer 完整源码学习手册

这份手册与当前仓库代码一一对应，目标不是只让你“能运行”，而是让你能够回答下面这些问题：

- 文本怎样变成 token id？
- 训练样本为什么要把输入和标签错开一个 token？
- 每一个张量在 Transformer 中的 shape 是什么？
- Multi-Head Self-Attention 的公式怎样对应 PyTorch 代码？
- causal mask 为什么在有 KV cache 时不再是普通的 `T × T` 方阵？
- Prefill 和 Decode 分别做了什么？为什么第一个生成 token 不需要 Decode？
- 为什么 cache 满后不能简单删除最旧的 K/V？
- 训练、普通前向、Prefill、Decode 四条执行路径有什么区别？
- 单元测试怎样证明缓存推理与完整前向等价？

建议一边阅读本文，一边打开已经生成的流程图：

- [完整模型流程图（SVG）](model_prefill_flowchart.svg)
- [模型主干与训练放大图（SVG）](model_training_detail.svg)
- [Attention 与 KV Cache 放大图（SVG）](attention_kv_cache_detail.svg)

---

## 1. 项目文件与执行顺序

| 文件 | 作用 | 建议阅读顺序 |
|---|---|---:|
| `tokenizer.py` | 构建词表，完成文本与 token id 的互相转换 | 1 |
| `train.py` | 构造 next-token 训练数据、训练模型并保存 checkpoint | 2 |
| `model.py` | Transformer、Attention、KV cache、Prefill 和 Decode 的核心实现 | 3 |
| `generate.py` | 加载 checkpoint，执行 Prefill + Decode 自回归生成 | 4 |
| `test_prefill.py` | 验证缓存推理、训练兼容性和滑窗控制流 | 5 |
| `data.txt` | 训练语料 | 随时查看 |

完整生命周期是：

```text
data.txt
   ↓
SimpleTokenizer 构建 stoi / itos
   ↓
连续 token ids
   ↓
随机截取 x 与右移一位的 targets
   ↓
TransformerLanguageModel 训练
   ↓
保存 model.pt（权重 + stoi + itos）
   ↓
generate.py 加载 checkpoint
   ↓
Prompt Prefill → 单 token Decode / 满窗 Re-Prefill
   ↓
生成文本
```

运行命令：

```bash
pip install -r requirements.txt
python train.py
python generate.py
python -m unittest -v test_prefill.py
```

`generate.py` 需要训练产生的 `model.pt`。当前仓库没有预置 checkpoint 时，应先运行训练。

---

## 2. 全项目统一符号

后面的公式和 shape 都使用下列符号：

| 符号 | 含义 | 当前配置 |
|---|---|---:|
| `B` | batch size | 训练时 16，当前生成时 1 |
| `T` | 本次前向新输入的 token 数 | 训练 8；Decode 为 1 |
| `P` | past KV cache 已保存的历史长度 | cache 状态可为 `0…8`；再次合法 Decode 前只能为 `0…7` |
| `S` | Attention 的总 key/value 长度，`S=P+T` | 最大 8 |
| `W` | 最大上下文窗口 `max_seq_len` | 8 |
| `V` | vocabulary size | 当前语料为 117 |
| `C` | `d_model`，每个 token 的隐藏维度 | 128 |
| `H` | attention head 数 | 4 |
| `D_h` | 每个 head 的维度，`C/H` | 32 |
| `L` | Transformer Block 数量 | 3 |
| `d_ff` | FFN 中间层维度，`4C` | 512 |

最重要的 shape 速查：

```text
input_ids                         [B, T]
token embedding                   [B, T, C]
position embedding                [T, C]      # 对 batch 广播
hidden state X                    [B, T, C]

Q, K_new, V_new（拆头后）          [B, H, T, D_h]
K_past, V_past                    [B, H, P, D_h]
K_all, V_all                      [B, H, S, D_h]
attention scores / weights        [B, H, T, S]
每个 head 的输出                   [B, H, T, D_h]
合并 head 后                       [B, T, C]

logits                            [B, T, V]
```

---

## 3. `tokenizer.py`：文本如何变成数字

### 3.1 为什么模型不能直接读取字符串

神经网络只能处理数值张量。Tokenizer 的任务是建立两张互逆的表：

- `stoi`：string to integer，例如 `"cat" -> 12`
- `itos`：integer to string，例如 `12 -> "cat"`

### 3.2 当前 Tokenizer 的处理规则

代码使用：

```python
words = text.lower().split()
```

这包含两步：

1. `lower()`：统一转为小写。
2. `split()`：按任意空白字符分词。

它是最简单的 word-level tokenizer，不是 BPE、WordPiece 或 SentencePiece。

例如：

```text
输入："The cat, sleeps."
结果：["the", "cat,", "sleeps."]
```

逗号和句号不会被单独拆开，所以 `cat` 与 `cat,` 会被视为不同 token。

### 3.3 词表的构造

```python
vocab = sorted(set(words))
self.itos = ["<pad>", "<unk>"] + vocab
```

- `set(words)` 去重。
- `sorted(...)` 让词表顺序稳定。
- `<pad>` 固定为 id 0。
- `<unk>` 固定为 id 1。

当前模型实际上没有 padding mask，也没有在 loss 中忽略 `<pad>`。因此 `<pad>` 目前只是被加入词表，并没有真正参与变长 batch 的 padding 机制。

### 3.4 `encode`

```python
self.stoi.get(word, self.unk_id)
```

如果词存在于词表，返回对应 id；否则返回 `<unk>` 的 id。

### 3.5 `decode`

```python
" ".join(self.itos[i] for i in ids)
```

它会用一个空格连接 token，因此不能还原：

- 原始大小写；
- 原始空格数量；
- 换行和制表符；
- 被 word-level 分词方式改变的标点结构。

这个 tokenizer 的价值在于帮助理解完整语言模型流程，而不是提供生产级文本编码质量。

---

## 4. `train.py`：Next-token Prediction 数据如何构造

### 4.1 连续 token 序列

整个 `data.txt` 被编码为一条一维序列：

```text
[token_0, token_1, token_2, ..., token_(N-1)]
```

当前数据约有 249 个 token。

### 4.2 输入和标签为什么错开一个 token

假设某个片段是：

```text
原序列：the  cat  sits  on  the  mat
```

构造：

```text
x:       the  cat  sits  on  the
targets: cat  sits  on    the mat
```

也就是：

```python
x = data[start:start + seq_len]
y = data[start + 1:start + seq_len + 1]
```

在位置 `t`：

- 模型输入已经看到 `x[0:t+1]`；
- `logits[:, t, :]` 要预测 `targets[:, t]`；
- `targets[:, t]` 正好是原序列中的下一个 token。

这就是自回归语言模型的 teacher forcing：训练时每个位置都使用真实历史 token，而不是模型自己刚生成的 token。

### 4.3 Batch shape

每次独立随机截取 `B=16` 个长度为 `T=8` 的片段，然后：

```python
x = torch.stack(xs)  # [16, 8]
y = torch.stack(ys)  # [16, 8]
```

最后将数据移动到 CPU 或 CUDA。

### 4.4 为什么随机窗口可以训练

虽然原始语料是一条长序列，但每次随机抽取不同的连续窗口，模型会逐渐看到语料中的不同局部上下文。这个项目没有 train/validation split，也没有 epoch 概念，而是直接执行固定的 5000 个随机更新 step。

### 4.5 模型和优化器

当前配置：

```text
batch_size   = 16
seq_len      = 8
d_model      = 128
num_heads    = 4
num_layers   = 3
learning_rate= 3e-4
train_steps  = 5000
```

优化器使用 AdamW。每一步执行：

```text
forward 得到 loss
→ optimizer.zero_grad() 清除上一步梯度
→ loss.backward() 反向传播并计算梯度
→ optimizer.step() 根据梯度更新参数
```

PyTorch 默认会累加 `.grad`。如果不调用 `zero_grad()`，当前 step 的梯度会与之前的梯度相加。

### 4.6 保存 checkpoint

checkpoint 包含：

```python
{
    "model_state_dict": model.state_dict(),
    "stoi": tokenizer.stoi,
    "itos": tokenizer.itos,
}
```

它没有保存模型超参数，因此 `generate.py` 必须用与训练完全相同的 `max_seq_len=8`、`d_model=128`、`num_heads=4`、`num_layers=3` 重建模型。

如果以后修改训练结构，建议同时把配置写入 checkpoint。

---

## 5. `model.py` 总览

模型是一个最小 decoder-only Transformer：

```text
input_ids [B,T]
    ↓
Token Embedding + Learned Absolute Position Embedding
    ↓
Transformer Block × 3
    ↓
Final LayerNorm
    ↓
LM Head: Linear(128, V)
    ↓
logits [B,T,V]
```

它没有：

- encoder；
- cross-attention；
- dropout；
- padding attention mask；
- RoPE 或 relative position；
- embedding 与 LM head 权重共享；
- EOS 自动停止机制。

---

## 6. Embedding 与位置编码

### 6.1 Token Embedding

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

可以把它理解成一个可训练查表矩阵：

```text
[V, C] = [117, 128]
```

每个 token id 取出一行，最终得到：

```text
token_emb [B,T,C]
```

### 6.2 Learned Absolute Position Embedding

Attention 本身不理解顺序。如果交换 token 顺序而不加入位置信息，Attention 只会看到一组向量。

当前代码使用可训练绝对位置表：

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```

没有 cache 时，位置是：

```text
0, 1, ..., T-1
```

有 cache 时，历史长度为 `P`，新 token 的位置必须从 `P` 开始：

```text
P, P+1, ..., P+T-1
```

因此代码是：

```python
positions = torch.arange(past_length, total_length)
```

位置 embedding 的 shape 是 `[T,C]`。与 `[B,T,C]` 的 token embedding 相加时，PyTorch 会沿 batch 维自动广播。

---

## 7. Multi-Head Self-Attention 逐行理解

### 7.1 为什么要有 Q、K、V

对每个 token 的隐藏向量 `Z`，分别经过三个独立线性层：

```text
Q = ZW_q + b_q   Query：当前 token 想查找什么
K = ZW_k + b_k   Key：每个 token 可以被怎样匹配
V = ZW_v + b_v   Value：匹配后真正汇总的信息
```

三个线性层输入输出都是 `[B,T,C]`。

### 7.2 拆成多个 head

当前 `C=128`、`H=4`，所以：

```text
D_h = C / H = 32
```

代码先：

```python
view(B, T, H, D_h)
```

再：

```python
transpose(1, 2)
```

得到：

```text
[B,H,T,D_h] = [B,4,T,32]
```

不同 head 可以学习关注不同类型的关系。

### 7.3 KV Cache 只缓存 K 和 V

如果已经处理过 `P` 个 token，每层的历史 cache 是：

```text
K_past [B,H,P,D_h]
V_past [B,H,P,D_h]
```

新 token 只计算：

```text
Q_new, K_new, V_new [B,H,T,D_h]
```

然后沿 sequence 维拼接：

```python
k = torch.cat([past_key, k], dim=2)
v = torch.cat([past_value, v], dim=2)
```

得到：

```text
K_all, V_all [B,H,S,D_h]
S = P + T
```

为什么不缓存 Q？

Q 只用于“当前 query 去查询所有 key”。历史 token 的 query 已经完成了自己的输出计算，未来不会再次使用；但历史 K/V 会被每个未来 query 反复读取，所以需要缓存。

### 7.4 Scaled Dot-Product Attention

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(self.head_dim)
```

shape：

```text
Q                         [B,H,T,D_h]
K^T                       [B,H,D_h,S]
QK^T                      [B,H,T,S]
```

除以 `sqrt(D_h)` 是为了避免维度较大时点积绝对值过大，导致 softmax 极端饱和、梯度变差。

### 7.5 有 cache 时为什么需要矩形 causal mask

普通 Prefill 没有 past，`P=0`。代码构造的布尔 mask 是 `T × T` 的严格上三角区域为 `True`，这些位置随后会被填成 `-inf`；从“允许访问”的角度看，未被屏蔽的区域才是熟悉的下三角结构。

有 cache 时：

- query 只有本次输入的 `T` 个；
- key 包含 `P` 个 past 加 `T` 个新 token；
- 所以 mask shape 是 `[T,S]`，不是 `[T,T]`。

第 `i` 个新 query 的全局位置是 `P+i`。它只能看 key 位置：

```text
0 ... P+i
```

不能看：

```text
P+i+1 ... S-1
```

因此屏蔽条件是：

```text
j > P+i
```

代码用：

```python
torch.triu(torch.ones(T, S), diagonal=P + 1)
```

例子：`P=3, T=2, S=5`：

```text
key index       0  1  2  3  4
new query 0     ✓  ✓  ✓  ✓  ×
new query 1     ✓  ✓  ✓  ✓  ✓
```

对于单 token Decode，`T=1, S=P+1`，这一行不会屏蔽任何 key：新 token 可以看见全部历史和自己。

### 7.6 Softmax 和 Value 汇总

```python
attention_weights = F.softmax(scores, dim=-1)
out = attention_weights @ v
```

softmax 沿 key 维进行，每个 query 对所有允许访问的 key 得到一组和为 1 的权重。

shape：

```text
weights                    [B,H,T,S]
V                          [B,H,S,D_h]
weights @ V                [B,H,T,D_h]
```

### 7.7 合并多个 head

```python
out.transpose(1, 2).contiguous().view(B, T, C)
```

把 `[B,H,T,D_h]` 变回 `[B,T,C]`，然后经过输出投影 `W_o`。

`contiguous()` 的作用是：`transpose` 只改变张量的 stride 视图，内存顺序可能不连续；`view` 通常要求内存布局可兼容，所以先转为连续内存。

---

## 8. Transformer Block：为什么叫 Pre-LN

当前 Block 的公式：

```text
Y       = X_l + MHA(LN1(X_l))
X_(l+1) = Y   + FFN(LN2(Y))
```

LayerNorm 在子层之前，所以称为 Pre-LN。

执行顺序：

```text
X_l
 ├─→ LayerNorm1 → Attention ─→ + ─→ Y
 └────────────────────────────↑

Y
 ├─→ LayerNorm2 → FFN ────────→ + ─→ X_(l+1)
 └────────────────────────────↑
```

两条跳线就是 residual connection。它让网络更容易保留原始信息，并改善深层网络的梯度传播。

### FFN

每个 token 独立经过：

```text
Linear(128, 512)
→ GELU
→ Linear(512, 128)
```

Attention 负责 token 之间的信息交互，FFN 负责对每个 token 的特征进行非线性变换。

---

## 9. `_forward_impl`：模型真正的统一执行核心

普通 forward、Prefill 和 Decode 最终都会复用 `_forward_impl`。

它依次做：

1. 检查 `input_ids` 必须是 `[B,T]`。
2. 检查 `T > 0`。
3. 校验每层 cache 的层数、shape、batch、head、head_dim 和历史长度。
4. 计算 `total_length = P + T`，保证不超过 `max_seq_len`。
5. 生成 cache-aware position ids。
6. Token Embedding 与 Position Embedding 相加。
7. 逐层传入对应的 `(K_l,V_l)`。
8. 如果 `use_cache=True`，收集每层更新后的 cache。
9. Final LayerNorm。
10. LM Head 得到 `[B,T,V]` logits。

KV cache 的完整结构是：

```text
(
    (K_layer_1, V_layer_1),
    (K_layer_2, V_layer_2),
    (K_layer_3, V_layer_3),
)
```

每一层都必须有自己的 K/V，因为每层输入隐藏状态和投影参数都不同。

---

## 10. 普通 `forward` 与训练 Loss

公共训练接口保持为：

```python
logits, loss = model(input_ids, targets)
```

它不使用 cache，以确保原训练代码保持简单。

如果提供 targets：

```python
logits.reshape(B * T, V)
targets.reshape(B * T)
```

交叉熵把 batch 和 sequence 两个维度合并，将每个位置看作一个 `V` 类分类问题。

数学形式：

```text
Loss = -(1 / (B·T)) Σ_(b,t) log softmax(logits[b,t])[targets[b,t]]
```

---

## 11. Prefill 与 Decode：最容易混淆的部分

### 11.1 Prefill

```python
logits, kv_cache = model.prefill(prompt_ids)
```

特征：

- `P=0`；
- 一次并行处理整个当前 prompt 窗口；
- 返回所有 prompt 位置的 logits；
- 同时为每一层建立 K/V cache。

假设 prompt 是：

```text
A B C
```

Prefill 返回：

```text
logits at A：预测 B 的分布
logits at B：预测 C 的分布
logits at C：预测第一个生成 token D 的分布
```

生成只需要：

```python
logits[:, -1, :]
```

### 11.2 为什么不能再 Decode prompt 的最后一个 token

Prefill 产生的 cache 已经包含 `A B C`。如果再次把 `C` 送入 Decode，cache 会错误地变成：

```text
A B C C
```

正确流程是直接使用 Prefill 最后位置的 logits 采样 `D`。

### 11.3 Decode

采样出 `D` 之后：

```python
logits_for_E, new_cache = model.decode(D, cache_ABC)
```

Decode 的输入是“刚生成的 token”，输出 logits 用于预测“再下一个 token”。

时间关系：

```text
Prefill(A,B,C) ─last logits→ sample D
Decode(D, cache_ABC) ─logits→ sample E
Decode(E, cache_ABCD) ─logits→ sample F
```

这就是常见的 off-by-one：Decode 的输入 token 和用它返回的 logits 所采样的 token不是同一个。

### 11.4 `@torch.no_grad()` 与 `model.eval()` 的区别

- `torch.no_grad()`：不构建反向传播图，减少显存和计算。
- `model.eval()`：让 Dropout、BatchNorm 等层切换到推理行为。

本项目没有 Dropout/BatchNorm，但 `load_runtime()` 仍然正确地调用了 `model.eval()`。Prefill/Decode 使用 `@torch.no_grad()`。

---

## 12. `generate.py` 完整控制流

### 12.1 懒加载 Runtime

模块导入时不会立刻读取 `model.pt`。第一次需要默认模型时，`_get_default_runtime()` 才调用 `load_runtime()`，然后缓存：

```text
(model, stoi, itos)
```

这样做的好处：

- `import generate` 不要求 checkpoint 一定存在；
- 单元测试可以注入 fake model；
- 多次 `generate()` 不会反复加载权重。

### 12.2 Prompt 编码

Prompt 经 lowercase + whitespace split 转为 ids，并放到模型参数所在 device。

### 12.3 初始窗口

```python
x_context = x[:, -model.max_seq_len:]
```

如果 prompt 超过 8 个 token：

- 模型只看到最后 8 个；
- 完整 `x` 不会被截断；
- 最终输出仍包含完整 prompt。

### 12.4 首次 Prefill

```python
logits, kv_cache = model.prefill(x_context)
```

之后从 `logits[:, -1, :]` 采样第一个新 token。

### 12.5 采样

```python
probs = torch.softmax(next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

这是 categorical random sampling：

- 不是 argmax；
- 没有 temperature；
- 没有 top-k；
- 没有 top-p；
- 没有屏蔽 `<pad>` 或 `<unk>`。

### 12.6 为什么先 append，再更新 cache

每一步先把采样 token 追加到完整输出 `x`。如果后面需要 Re-Prefill，最新窗口必须包含刚生成的 token。

### 12.7 Cache 未满

```python
logits, kv_cache = model.decode(next_token, kv_cache)
cached_length += 1
```

只计算新 token，复用历史 K/V。

`cached_length` 表示“已经被模型消费并写入 cache 的 token 数”。刚刚 sample 并追加到完整输出 `x`、但尚未执行 Decode 的 token，还没有计入 cache；只有 Decode 完成或 Re-Prefill 完成后，cache 才包含它。

### 12.8 Cache 已满为什么必须 Re-Prefill

当 cache 长度达到 8，再调用 Decode 会得到总长度 9，超过位置 embedding 范围。

不能简单删除最旧 K/V，原因有两个：

1. 当前使用 learned absolute position。窗口滑动后，保留 token 应从旧位置 `1…7` 重标为新位置 `0…6`。
2. 深层 token 表征已经通过 Attention 融合了被删除的旧 token。即使能修改位置，直接保留深层 K/V 也不等价于“只对最新窗口重新前向”。

因此代码：

```python
x_context = x[:, -model.max_seq_len:]
logits, kv_cache = model.prefill(x_context)
```

cache 一旦满 8，之后每个非最终生成 step 都会重新 Prefill 最新 8 个 token。

### 12.9 边界行为

- `max_new_tokens < 0`：报错。
- `max_new_tokens == 0`：直接返回 prompt 的 tokenizer 解码结果，不执行 `forward/prefill/decode`。但在没有注入 `model/stoi/itos` 时，代码仍会先加载默认 checkpoint、构造模型并确定设备；“不执行模型前向”不等于“完全不加载模型”。
- 需要生成但 prompt 为空：报错，因为项目没有 BOS token。
- 没有 EOS，所以不会提前停止。
- 最后一个 token 采样后直接结束，不再进行一次无用 Decode。

---

## 13. 一次完整的 shape 跟踪

### 13.1 Prefill：`B=1, T=3, P=0`

```text
input_ids                         [1,3]
token_emb                         [1,3,128]
position_emb                      [3,128]
X                                 [1,3,128]

Q,K_new,V_new                     [1,4,3,32]
K_all,V_all                       [1,4,3,32]
scores                            [1,4,3,3]
weights                           [1,4,3,3]
attention output                  [1,3,128]

每层 cache K/V                    [1,4,3,32]
logits                            [1,3,117]
```

### 13.2 Decode：`B=1, T=1, P=3, S=4`

```text
new input_id                      [1,1]
position id                       [3]
new hidden                        [1,1,128]

Q_new,K_new,V_new                 [1,4,1,32]
K_past,V_past                     [1,4,3,32]
K_all,V_all                       [1,4,4,32]
scores                            [1,4,1,4]
mask                              [1,4]，全部允许
attention output                  [1,1,128]

new per-layer cache K/V           [1,4,4,32]
logits                            [1,1,117]
```

---

## 14. 参数量怎样得到 626,165

这个数字依赖当前 `data.txt` 构造出的 `V=117`；换语料后词表大小和总参数量都会变化。

### Embedding

```text
Token embedding:     117 × 128 = 14,976
Position embedding:    8 × 128 =  1,024
```

### 每个 Block

Attention 有 `W_q/W_k/W_v/W_o` 四个 `128 → 128` Linear，均有 bias：

```text
4 × (128×128 + 128) = 66,048
```

FFN：

```text
(128×512 + 512) + (512×128 + 128) = 131,712
```

两个 LayerNorm，每个有 weight 和 bias：

```text
2 × (128 + 128) = 512
```

单 Block：

```text
66,048 + 131,712 + 512 = 198,272
```

三个 Block：

```text
594,816
```

最后：

```text
Final LayerNorm:              256
LM Head: 128×117 + 117 =   15,093
```

总计：

```text
14,976 + 1,024 + 594,816 + 256 + 15,093 = 626,165
```

用符号表示，在所有 Linear 都含 bias、LayerNorm 含 weight/bias 的当前实现中：

```text
每个 Block = 12C² + 13C
总参数量    = VC + WC + L(12C² + 13C) + 2C + CV + V
```

其中前两项是 token/position embedding，中间是 `L` 个 Block，`2C` 是 Final LayerNorm，最后两项是 LM Head。

---

## 15. 复杂度与 Cache 内存

忽略常数，先看单层：

### Prefill

```text
O(T·C² + T²·C)
```

- 线性投影和 FFN 主要是 `T·C²`。
- Attention 的 query-key/value 交互主要是 `T²·C`。

### 单 token Decode

```text
O(C² + S·C)
```

不再为历史 token 重算 QKV 和 FFN，但新 query 仍要读取全部 `S` 个 K/V。

若计入 `L` 层和最终词表投影，完整模型的主要时间复杂度约为：

```text
Prefill:           O(L·(T·C² + T²·C) + T·C·V)
单 token Decode:   O(L·(C² + S·C) + C·V)
```

但当前 `generate.py` 使用 learned absolute position，并且 cache 满后会对最新的 `W` 个 token Re-Prefill。此后每个仍需继续生成的 step，成本会回到：

```text
O(L·(W·C² + W²·C) + W·C·V)
```

因此“单 token Decode 的加速”只发生在 cache 尚未填满的阶段；当前窗口很小（`W=8`），实现重点是语义清晰，而非长上下文服务性能。

### Cache 大小

每层 K 和 V：

```text
2 × B × H × S × D_h
```

所有层：

```text
2 × L × B × S × C
```

当前 `L=3, B=1, S=8, C=128`：

```text
2 × 3 × 1 × 8 × 128 = 6,144 floats
```

FP32 约为 24 KiB。

这个数字只统计 K/V 元素，不包含模型参数、logits、Attention 临时矩阵、Python 容器和内存分配器开销；如果 cache dtype 改成 FP16/BF16，纯 K/V 字节数会减半。

注意当前代码使用 `torch.cat` 增长 cache，每一步都会分配新张量并复制旧 cache，约产生 `O(L·B·S·C)` 的 cache 内存读写。生产级服务通常会预分配 cache、使用 paged KV cache 或专用 kernel。

---

## 16. `test_prefill.py` 在证明什么

### 16.1 `test_prefill_matches_regular_forward`

同一个输入分别执行普通 forward 和 Prefill，所有 logits 必须接近相等。

它证明建立 cache 没有改变模型计算结果；测试还断言 cache 的层数，以及每层 K/V 的 `[B,H,T,D_h]` shape。

### 16.2 `test_cached_decode_matches_regular_forward`

先对前 3 个 token Prefill，再逐 token Decode；每步 Decode 的 logits 与一次性完整 forward 对应位置的 logits 比较。

它能发现：

- position offset 错误；
- cache 拼接维度错误；
- 会错误屏蔽历史 key 或当前 token 的 mask 错误；
- 最终 cache 长度没有随 Decode 正确增长的问题。

该测试每次 Decode 都是 `T=1`，因此没有覆盖 `P>0,T>1` 的“分块缓存前向”矩形 mask。当前公开 `decode()` 本来就只接受一个 token；如果以后增加 chunked decode，应额外加入 `P>0,T>1` 的数值等价测试。

### 16.3 `test_training_forward_contract_is_unchanged`

检查：

- `model(x, targets)` 仍然返回二元组；
- logits shape 正确；
- loss 存在；
- `loss.backward()` 能产生梯度。

### 16.4 非法输入测试

覆盖：

- 非法 `max_seq_len`；
- 0 层模型；
- 空 Prefill；
- cache 溢出；
- Decode 输入不是 `[B,1]`。

### 16.5 `RecordingModel`

Fake model 会：

- 固定生成 token 0，使测试完全确定；
- 记录每次调用是 Prefill 还是 Decode；
- 记录每次输入的 token。

因此生成测试可以精确检查：

- 初始 Prompt 只 Prefill 一次；
- 后续 Decode 的确只收到刚生成的 token；
- cache 满时走 Re-Prefill；
- 长 prompt 只 Prefill 最近窗口；
- 生成 0 个 token 时模型完全不被调用。

其中 `test_generate_prefills_once_then_decodes_new_tokens` 还会检查第一次 Decode 收到的是“刚采样的新 token”，而不是重复收到 prompt 的最后一个 token，因此它专门捕获生成循环的 off-by-one 错误。

---

## 17. 最常见的理解错误

### 错误 1：Prefill 后还要 Decode prompt 最后一个 token

错误。Prefill cache 已包含整个 prompt；第一个 token 直接由最后位置 logits 采样。

### 错误 2：KV cache 缓存模型最终 hidden state

错误。每层缓存的是该层 Attention 投影后的 K/V，并且每层各不相同。

### 错误 3：有 cache 后 causal mask 仍然是 `T × T`

错误。它必须是 `[T,P+T]`，否则新 query 无法正确访问历史 key。

### 错误 4：Decode 的输出 logits 用来预测它自己的输入 token

错误。`decode(D)` 的 logits 用来预测 `D` 后面的 token。

### 错误 5：Cache 满时直接删掉最左侧 K/V 即可

对当前 learned absolute position 模型不成立，而且深层表示仍融合了被删上下文。

### 错误 6：`model.eval()` 会自动关闭梯度

错误。`eval()` 和 autograd 是两个独立机制；应配合 `no_grad()` 或 `inference_mode()`。

### 错误 7：Softmax 之后一定选择概率最大的 token

错误。当前代码使用 `multinomial` 随机采样。

---

## 18. 建议的调试方式

### 18.1 打印 shape

建议第一次学习时在 Attention 中临时打印：

```python
print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)
print("scores:", scores.shape)
print("mask:", mask.shape)
print("attention_weights:", attention_weights.shape)
```

先使用很小的 batch 和训练步数，避免输出过多。

### 18.2 手工打印 mask

分别尝试：

```text
P=0,T=4
P=3,T=1
P=3,T=2
```

确认每行允许访问的 key 与公式 `j ≤ P+i` 一致。

### 18.3 固定随机种子

训练和采样都包含随机性。调试时可以设置：

```python
random.seed(0)
torch.manual_seed(0)
```

CUDA 还可能需要额外的确定性设置。

### 18.4 用测试理解不变量

最重要的不变量：

```text
完整 forward 对应位置 logits
≈ Prefill(prefix) + 逐 token Decode 得到的 logits
```

如果不相等，优先检查 position、mask 和 cache 拼接。

---

## 19. 推荐学习与改造练习

按顺序完成，能够帮助你真正掌握项目。

### Level 1：基础理解

1. 手算一个长度为 4 的 `x/targets`。
2. 打印 tokenizer 的 `stoi/itos`。
3. 打印模型每一层的 shape。
4. 手画 `T=4` 的 causal mask。

### Level 2：生成策略

1. 添加 greedy decoding（`argmax`）。
2. 添加 temperature。
3. 添加 top-k sampling。
4. 比较随机采样与 greedy 输出。

### Level 3：训练能力

1. 添加 validation split。
2. 定期计算 validation loss。
3. 将模型超参数保存进 checkpoint。
4. 添加随机种子和更完整的日志。

### Level 4：Tokenizer 和 Batch

1. 将标点单独分词。
2. 添加 BOS/EOS。
3. 添加 padding mask。
4. 在 loss 中使用 `ignore_index=pad_id`。

### Level 5：高效推理

1. 预分配固定大小 KV cache，避免每步 `torch.cat`。
2. 尝试 RoPE 或相对位置编码。
3. 实现真正的滑动窗口 cache。
4. 支持 batch 中不同 prompt 长度。
5. 对比普通生成和 cached generation 的耗时。

---

## 20. 面试式自检问题

读完后，尝试不看答案直接回答：

1. 为什么 `d_model` 必须能被 `num_heads` 整除？
2. Q/K/V 为什么要先 reshape 再 transpose？
3. 为什么 Attention score 要除以 `sqrt(head_dim)`？
4. 有 cache 时，mask 为什么是矩形？
5. 为什么 mask 的 diagonal 是 `P+1`？
6. 为什么只缓存 K/V，不缓存 Q？
7. Prefill 返回全部 logits，为什么生成只使用最后一个？
8. 为什么不能把 prompt 最后一个 token 再送入 Decode？
9. `decode(y_k)` 返回的 logits 预测哪个 token？
10. 为什么每个 Transformer 层都需要独立 cache？
11. 为什么 cache 满后不能简单 pop 最旧项？
12. `model.eval()` 与 `torch.no_grad()` 有什么区别？
13. 当前生成为什么不会提前停止？
14. 当前 batch 为什么不能安全包含 padding？
15. 单 token Decode 相比整段重算节省了什么，仍然保留了什么开销？

如果这些问题都能结合代码和 shape 清楚回答，你已经掌握了这份项目最核心的知识。

