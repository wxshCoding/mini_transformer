"""训练这个最小 Transformer 自回归语言模型。

运行本文件时，完整的数据流如下：

``data.txt 原始文本``
    -> ``SimpleTokenizer`` 构建词表
    -> 整篇文本编码为一维 token id 序列
    -> 随机截取多个长度为 ``seq_len`` 的滑动窗口
    -> 将窗口右移一位作为 next-token 监督标签
    -> Transformer 输出每个位置对整个词表的 logits
    -> 交叉熵损失比较 logits 和正确的下一个 token
    -> 反向传播并用 AdamW 更新参数
    -> 将模型权重与 tokenizer 词表保存到 ``model.pt``

这是一个有意保持简洁的教学训练脚本。它没有划分验证集，也没有学习率调度、
梯度裁剪、混合精度、早停或断点续训逻辑，便于初学者先看清一次标准训练迭代
最核心的步骤。
"""

import random

import torch

from tokenizer import SimpleTokenizer
from model import TransformerLanguageModel


# ---------------------------------------------------------------------------
# 1. 训练超参数（hyperparameters）
# ---------------------------------------------------------------------------
# 超参数不是模型通过训练自动学习的权重，而是训练开始前由开发者设定的配置。

# 一个 batch 中并行训练的样本数。这里每个样本都是从整篇语料随机截取的一段。
# batch 越大，单步梯度通常越稳定，但会占用更多显存/内存。
batch_size = 16

# 每个训练样本包含的 token 数，也就是模型本次配置下的上下文窗口长度 T。
# 输入 x 和目标 y 的形状都会是 [batch_size, seq_len] = [16, 8]。
seq_len = 8

# 每个 token 在模型内部使用的向量维度 C。token embedding、位置 embedding、
# 每个 Transformer Block 的输入输出，以及最终 LayerNorm 都使用这个维度。
d_model = 128

# 多头注意力的头数 H。每个头的维度为 head_dim = d_model / num_heads = 32。
# 因此 d_model 必须能被 num_heads 整除，模型代码中用 assert 检查这一条件。
num_heads = 4

# 串联的 Transformer Block 数量，也可理解为网络深度。层数更多通常带来更强
# 表达能力，同时增加参数量、计算量和 KV cache 占用。
num_layers = 3

# AdamW 每次更新参数时使用的学习率。过大会导致训练不稳定，过小会收敛很慢。
# 3e-4 等价于 0.0003，是训练小型 Transformer 时常见的起始量级。
learning_rate = 3e-4

# 总优化步数。每一步都会重新随机采样一个 batch，并执行一次参数更新。
train_steps = 5000


# ---------------------------------------------------------------------------
# 2. 选择计算设备
# ---------------------------------------------------------------------------
# torch.cuda.is_available() 检查当前 PyTorch 是否能使用 NVIDIA CUDA GPU。
# 若可用，device 为 "cuda"；否则回退到 "cpu"。随后模型和每个 batch 都要被
# 放到同一 device 上，否则张量运算会报“设备不一致”的错误。
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# ---------------------------------------------------------------------------
# 3. 读取训练语料并建立 tokenizer
# ---------------------------------------------------------------------------
# encoding="utf-8" 使 Python 按 UTF-8 解码文件，可正确读取常见中英文文本。
# with 代码块结束时文件会自动关闭，即使读取过程中发生异常也无需手动 close。
with open("data.txt", "r", encoding="utf-8") as f:
    # 一次性读入完整文件。该做法适合本项目的小语料；大规模数据通常需要分块
    # 或流式加载，避免把全部文本同时放入内存。
    text = f.read()


# tokenizer 扫描整份训练文本：转小写、按空白切词、去重排序，并在普通单词前
# 加入 <pad> 与 <unk>。训练和推理必须使用完全相同的 stoi/itos 映射。
tokenizer = SimpleTokenizer(text)

print("vocab size:", tokenizer.vocab_size)


# ---------------------------------------------------------------------------
# 4. 把整篇语料转换为一维 token id 张量
# ---------------------------------------------------------------------------
# encode 返回 Python 整数列表。假设语料分词后共有 N 个 token，那么 data 此时
# 类似 [12, 7, 45, ...]，长度为 N；相邻元素仍保持原文中的先后顺序。
data = tokenizer.encode(text)

# nn.Embedding 的索引和 cross_entropy 的类别标签都要求整数类型；PyTorch 中
# torch.long 通常是 64 位有符号整数（int64），正是这两个算子期望的 dtype。
# data 的形状为 [N]，并且此时仍在 CPU 上；仅将采样后的 batch 搬到 device，
# 避免为了这个教学脚本把整份语料常驻显存。
data = torch.tensor(data, dtype=torch.long)

print("number of tokens:", len(data))


# ---------------------------------------------------------------------------
# 5. 随机采样一个 next-token prediction batch
# ---------------------------------------------------------------------------
def get_batch():
    """从一维语料中随机截取并组装一个训练 batch。

    自回归语言模型的目标是：根据当前位置及其之前的 token，预测下一个 token。
    如果原始片段为：

    ``[t0, t1, t2, t3, ..., t8]``

    当 ``seq_len == 8`` 时，会构造：

    * 输入 ``x = [t0, t1, t2, ..., t7]``
    * 目标 ``y = [t1, t2, t3, ..., t8]``

    可见 y 相对 x 整体向右移动一个 token。于是 x 的第 j 个位置要预测 y 的
    第 j 个位置，也就是原始序列中的下一个 token。模型内部的 causal mask
    保证每个位置不能偷看未来输入。

    返回
    ----
    x:
        输入 token id，形状 ``[B, T]``，这里为 ``[16, 8]``。
    y:
        每个输入位置对应的下一个 token id，形状同样为 ``[B, T]``。

    两个张量返回前都会移动到前面选定的 ``device``。
    """

    # 先用列表收集 batch 中每一条一维样本，最后再 stack 成二维张量。
    xs = []
    ys = []

    # 循环 batch_size 次，相互独立地随机选取 batch 中的每个训练样本。
    # 这是“有放回采样”：不同样本偶尔可能选择到相同的 start。
    for _ in range(batch_size):
        # random.randint(a, b) 的两端都可能取到。上界确保后面截取 y 时仍有
        # 足够的 token。当前写法 len(data) - seq_len - 2 还额外保守地留出
        # 一个 token，不会越过 data 尾部。
        #
        # 因此语料必须明显长于 seq_len；若上界小于 0，randint 会报错。
        # 本脚本没有固定 random seed，所以不同运行采到的 batch 顺序会不同。
        start = random.randint(
            0,
            len(data) - seq_len - 2
        )

        # 从 start 起连续读取 T=seq_len 个 token，形状都是 [T]。
        x = data[start:start + seq_len]

        # 起点和终点都向右移动 1，因此 y[j] 正好等于 x[j] 后面的 token。
        # x 和 y 在语料上大量重叠，这就是 next-token 监督信号的来源。
        y = data[start + 1:start + seq_len + 1]

        xs.append(x)
        ys.append(y)

    # 每条 x/y 的形状为 [T]；torch.stack 默认沿新建的第 0 维堆叠，得到
    # [B, T]。注意 stack 要求列表中所有张量形状完全相同。
    x = torch.stack(xs)
    y = torch.stack(ys)

    # 模型参数已经位于 device，因此输入和标签也必须移到同一设备。
    # CPU 情况下 .to(cpu) 基本保持原状；CUDA 情况下这里发生 CPU -> GPU 拷贝。
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 6. 实例化 Transformer 语言模型
# ---------------------------------------------------------------------------
model = TransformerLanguageModel(
    # token embedding 需要知道可接受多少种 id，lm_head 也需要为词表中的每个
    # token 输出一个分数，所以两端都使用 tokenizer.vocab_size。
    vocab_size=tokenizer.vocab_size,

    # 模型使用可学习的位置 embedding，其容量由 max_seq_len 决定。这里将其
    # 设为训练窗口长度，因此单次完整 forward/prefill 最多处理 8 个位置。
    max_seq_len=seq_len,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)
# .to(device) 会把模型中所有参数和 buffer 搬到所选设备，并返回模型本身。


# model.parameters() 迭代模型注册的所有参数张量；p.numel() 返回一个张量中的
# 标量元素个数。求和得到模型总参数量，便于了解模型规模和大致内存开销。
# 这里统计的是全部参数；当前模型中的参数默认都 requires_grad=True、可训练。
num_parameters = sum(
    p.numel() for p in model.parameters()
)

print("parameters:", num_parameters)


# ---------------------------------------------------------------------------
# 7. 创建 AdamW 优化器
# ---------------------------------------------------------------------------
# 优化器持有所有可训练参数的引用，并在 optimizer.step() 时根据参数的梯度更新
# 权重。AdamW 为每个参数维护一阶、二阶动量，并将 weight decay 与梯度更新解耦；
# 它是 Transformer 中常用的优化器。
#
# 此处只显式设置学习率，其余参数（如 betas、eps、weight_decay）使用当前
# PyTorch AdamW 的默认值。优化器状态未在最后的 checkpoint 中保存。
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


# ---------------------------------------------------------------------------
# 8. 主训练循环
# ---------------------------------------------------------------------------
# train() 将模块切换到训练模式。它不会自行执行训练，也不会开启梯度；它主要
# 影响 Dropout、BatchNorm 等训练/推理行为。当前模型虽没有 Dropout/BatchNorm，
# 显式调用仍是规范做法，也方便未来扩展网络。
model.train()

for step in range(train_steps):
    # (1) 每一步重新随机采样数据：x、y 形状都是 [B, T]，dtype 为 long。
    x, y = get_batch()

    # (2) 前向传播。
    # logits 形状为 [B, T, V]：每个样本、每个序列位置，都得到 V=vocab_size
    # 个未归一化分数。因为传入了 y，model.forward 还会把 [B,T,V] 和 [B,T]
    # 展平后计算交叉熵，返回标量 loss（整个 batch、全部位置的平均损失）。
    logits, loss = model(x, y)

    # (3) 清除上一步遗留的梯度。
    # PyTorch 默认会把每次 backward 得到的梯度累加到 parameter.grad；如果不
    # 清零，本步梯度就会错误地与之前所有步骤相加。梯度累积训练会有意推迟
    # 清零，但本脚本每个 batch 都独立更新一次，所以每步先 zero_grad()。
    optimizer.zero_grad()

    # (4) 反向传播。
    # autograd 从标量 loss 沿前向计算图应用链式法则，为所有可训练参数计算
    # d(loss)/d(parameter)，并把结果写入各参数的 .grad 属性。
    loss.backward()

    # (5) 参数更新。
    # AdamW 读取刚才生成的 .grad 以及自身保存的动量状态，按 learning_rate
    # 更新模型参数。执行完后，新权重会在下一次前向传播中生效。
    optimizer.step()

    # 每 100 步打印一次当前 batch 的损失。step 从 0 开始，所以训练第一步
    # 也会打印。loss.item() 将只含一个元素的张量转换为 Python 浮点数；
    # ``:5d`` 让步数右对齐，``:.4f`` 将损失显示为小数点后 4 位。
    if step % 100 == 0:
        print(
            f"step={step:5d} "
            f"loss={loss.item():.4f}"
        )


# ---------------------------------------------------------------------------
# 9. 保存训练结果（checkpoint）
# ---------------------------------------------------------------------------
# torch.save 会把下面的字典序列化到 model.pt：
#
# * model_state_dict：参数名 -> 权重张量，是恢复模型权重的标准形式；
# * stoi / itos：训练时的双向词表，推理时必须复用，不能根据新 prompt 重建。
#
# 这个 checkpoint 没有保存模型超参数、当前 step、优化器状态或随机数状态，
# 因而足够供本项目的生成脚本加载，但不能精确地从中断处恢复完整训练过程。
# 加载时还需先用相同 d_model、num_heads、num_layers、max_seq_len 实例化模型，
# 再调用 load_state_dict(...) 加载权重。
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    },
    "model.pt",
)

print("model saved")
