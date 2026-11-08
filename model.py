"""一个带 KV Cache 的最小 Decoder-only Transformer 语言模型。

本文件包含模型本身的全部核心组件：多头因果自注意力、前馈网络、
Pre-LayerNorm Transformer Block、训练前向，以及推理时的 Prefill/Decode。

全文统一使用以下张量维度符号：

    B：batch size
    T：本次调用中新输入的 token 数量（current/query sequence length）
    P：KV Cache 中已经保存的历史 token 数量（past length）
    S：注意力可见的 key/value 总长度，S = P + T
    C：模型隐藏维度 d_model
    H：注意力头数 num_heads
    D：每个头的维度 head_dim = C / H
    V：词表大小 vocab_size
    L：Transformer Block 层数 num_layers

主要张量形状：

    input_ids / targets       [B, T]
    hidden states             [B, T, C]
    Q（拆分多头后）            [B, H, T, D]
    K、V（拼接历史后）          [B, H, S, D]
    attention scores/weights  [B, H, T, S]
    logits                    [B, T, V]
    单层 cache 的 key/value     [B, H, P 或 S, D]

这是教学用精简实现：没有 padding mask、dropout、RoPE 和权重绑定。
批量输入中的每个位置都会被当作真实 token；若加入 padding，需要同时扩展
注意力掩码，避免模型关注 padding 位置。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """带因果掩码和可选 KV Cache 的多头自注意力。

    “Self-Attention”表示 Q、K、V 均由当前层的同一份输入 ``x`` 投影得到。
    在增量推理时，当前输入只生成当前 token 的 Q/K/V，历史 K/V 则从
    ``past_key_value`` 读取，并与当前 K/V 沿序列维拼接。

    参数:
        d_model: 模型隐藏维度 C；输入、输出最后一维均为 C。
        num_heads: 注意力头数 H。它必须整除 d_model，因此每头维度
            D = d_model // num_heads。

    数学形式（省略 batch 和 head 下标）::

        Q = X W_Q,  K = X W_K,  V = X W_V
        A = softmax(Q K^T / sqrt(D) + M)
        Y = A V W_O

    因果掩码 M 在 key 位于 query 的未来时取负无穷，否则取 0；所以
    softmax 后，未来位置的注意力概率为 0。
    """

    def __init__(self, d_model, num_heads):
        """创建 Q/K/V/O 四个投影层，并计算每个注意力头的维度。"""
        super().__init__()

        # 多头拆分要求 C = H * D，否则最后一维无法平均分给 H 个头。
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # nn.Linear 作用于张量最后一维。PyTorch 的具体计算为
        # y = x @ weight.T + bias；这里四个投影都保留 bias。
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        """计算当前 token 的多头因果自注意力。

        参数:
            x: 当前层的新输入，形状 ``[B,T,C]``。普通训练时 T 是完整
                序列长度；Prefill 时 T 是 prompt 长度；Decode 时 T=1。
            past_key_value: 可选 ``(past_key, past_value)``。两个张量形状
                均为 ``[B,H,P,D]``，保存“本层”前 P 个 token 的 K/V。
            use_cache: 是否把拼接当前 token 后的完整 K/V 一并返回。
                它只控制返回值；只要传入 ``past_key_value``，注意力就会
                使用历史缓存，不论该参数是 True 还是 False。

        返回:
            ``use_cache=False`` 时返回 ``out [B,T,C]``；
            ``use_cache=True`` 时返回 ``(out, (key, value))``，其中新的
            key/value 形状均为 ``[B,H,P+T,D]``。

        因果掩码的绝对位置语义:
            当前局部 query i 在完整上下文中的位置为 P+i，它只能看到
            ``j <= P+i`` 的 key。实现中的 ``diagonal=P+1`` 正是在把
            ``j > P+i`` 的未来位置屏蔽掉。
        """
        # x: [batch, current_sequence, hidden]。
        # 正常情况下 C == self.d_model，否则下面的线性投影会报维度错误。
        B, T, C = x.shape

        # 在尚未拆头的 C 维空间生成 Q/K/V，三者均为 [B,T,C]。
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # [B,T,C] -> [B,T,H,D] -> [B,H,T,D]。
        # 把 H 移到序列维前面，随后即可并行执行各个头的矩阵乘法。
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 没有历史 cache 时 P=0，对应常规训练或从头开始的 Prefill。
        past_length = 0

        if past_key_value is not None:
            # past_key/past_value: [B,H,P,D]。
            # 不缓存 Q：历史 query 用过即完成使命；未来 token 只需要用自己
            # 的新 Q 查询历史 K，再用注意力权重聚合历史 V。
            past_key, past_value = past_key_value
            past_length = past_key.size(2)

            # 防止例如 CPU cache 与 CUDA 输入混用。模型级校验负责形状，
            # 此处用当前投影结果检查运行设备，错误信息比 torch.cat 更明确。
            if past_key.device != k.device or past_value.device != v.device:
                raise ValueError("Cached keys and values must be on the same device as the input")

            # 混合精度下也必须同步转换 cache；float32 历史不能直接与
            # float16 当前 K/V 拼接。
            if past_key.dtype != k.dtype or past_value.dtype != v.dtype:
                raise ValueError("Cached keys and values must have the same dtype as the input")

            # dim=2 是序列维：[B,H,P,D] 与 [B,H,T,D]
            # 拼为 [B,H,P+T,D]。torch.cat 会创建新张量，不会原地修改旧 cache。
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        # [B,H,T,D] @ [B,H,D,P+T] -> [B,H,T,P+T]。
        # scores[b,h,i,j] 是 query i 与 key j 的未归一化点积相似度。
        scores = q @ k.transpose(-2, -1)

        # Scaled Dot-Product Attention：除以 sqrt(D) 可防止 D 较大时点积
        # 方差过大、softmax 过早饱和，从而改善数值稳定性和梯度。
        scores = scores / math.sqrt(self.head_dim)

        # 构造二维布尔掩码 [T,P+T]；masked_fill 时会自动广播到
        # [B,H,T,P+T]，所以无需为每个 batch/head 复制一份。
        #
        # torch.triu(..., diagonal=d) 在 j-i >= d 时为 True。令 d=P+1：
        #     j-i >= P+1  <=>  j > P+i
        # 这恰好屏蔽 query i 的未来 key。
        #
        # P=0：退化为普通上三角 causal mask，query i 只看 0..i。
        # P=3,T=1：mask 为 [1,4] 且全 False；绝对位置 3 的新 query
        #           可以看历史 0..2 和自己 3。
        # P=3,T=2：两个新 query 分别可看 0..3 与 0..4。
        mask = torch.triu(
            torch.ones(
                T,
                k.size(2),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=past_length + 1,
        )

        # 被屏蔽分数写成 -inf；softmax 后对应概率严格为 0。
        scores = scores.masked_fill(mask, float("-inf"))

        # 沿全部 key 的维度归一化。每个样本、每个头、每个 query 的
        # 权重总和为 1；形状仍为 [B,H,T,P+T]。
        attention_weights = F.softmax(scores, dim=-1)

        # [B,H,T,P+T] @ [B,H,P+T,D] -> [B,H,T,D]：
        # 使用注意力概率对全部可见 value 做加权求和。
        out = attention_weights @ v

        # [B,H,T,D] -> [B,T,H,D] -> [B,T,C]。transpose 得到的通常是
        # 非连续视图，因此 view 前必须调用 contiguous()。
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影在 C 维混合各头的信息，张量形状保持 [B,T,C]。
        out = self.W_o(out)

        if use_cache:
            # k/v 已包含 past 与 current，可直接传给下一次增量 Decode。
            return out, (k, v)

        return out


class FeedForward(nn.Module):
    """对每个 token 独立应用的两层前馈网络（FFN）。

    参数:
        d_model: 输入和输出隐藏维度 C。

    公式::

        FFN(x) = W_2 GELU(W_1 x + b_1) + b_2

    第一层把 C 扩展到 4C，提供更高维的非线性特征空间；第二层投影回 C，
    以便和残差分支相加。线性层只作用于最后一维，所以 FFN 不在不同 token
    之间交换信息；token 间的信息交换发生在 Self-Attention 中。
    """

    def __init__(self, d_model):
        """创建 ``C -> 4C -> C`` 的前馈子网络。"""
        super().__init__()

        self.net = nn.Sequential(
            # [B,T,C] -> [B,T,4C]
            nn.Linear(d_model, 4 * d_model),
            # GELU 是 Transformer 常用的平滑非线性函数，形状不变。
            nn.GELU(),
            # [B,T,4C] -> [B,T,C]
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        """逐 token 变换 ``x [B,T,C]``，返回同形状的张量。"""
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个采用 Pre-LayerNorm 的 Decoder Transformer Block。

    参数:
        d_model: 隐藏维度 C。
        num_heads: 自注意力头数 H。

    块内计算顺序::

        a = x + Attention(LN_1(x))
        y = a + FFN(LN_2(a))

    LayerNorm 位于子层之前，所以叫 Pre-LN；两个加号都是残差连接，为信息
    与梯度提供直接路径。Attention/FFN 的输出必须保持 ``[B,T,C]``，才能
    与残差分支逐元素相加。

    每层都有独立 KV Cache：第 l 层的 K/V 来自第 l 层 ``norm1(x)`` 的
    投影，包含该层特有的表示，不能拿到另一层使用。
    """

    def __init__(self, d_model, num_heads):
        """创建两个 LayerNorm、一个注意力子层和一个 FFN 子层。"""
        super().__init__()

        # LayerNorm 在最后一个 C 维上，对每个样本、每个 token 独立归一化。
        self.norm1 = nn.LayerNorm(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        """执行一个 Pre-LN Transformer Block。

        参数:
            x: 当前 token 的块输入，形状 ``[B,T,C]``。
            past_key_value: 本层的历史 ``(key,value)``，各为
                ``[B,H,P,D]``；None 表示无历史缓存。
            use_cache: 是否返回本层更新后的 K/V。

        返回:
            ``use_cache=False`` 时返回块输出 ``[B,T,C]``；
            ``use_cache=True`` 时返回 ``(x, present_key_value)``，其中
            present K/V 均为 ``[B,H,P+T,D]``。
        """
        # Pre-LN：先归一化，再进入注意力。返回结构由 use_cache 决定。
        attention_output = self.attention(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if use_cache:
            # 缓存开启时拆出注意力结果与本层更新后的 (K,V)。
            attention_output, present_key_value = attention_output

        # 第一条残差：a = x + Attention(LN1(x))。
        x = x + attention_output

        # 第二条残差：y = a + FFN(LN2(a))。norm2 的输入是已经融合了
        # 注意力结果的 a，而不是最初进入 Block 的 x。
        x = x + self.ffn(self.norm2(x))

        if use_cache:
            return x, present_key_value

        return x


class TransformerLanguageModel(nn.Module):
    """Decoder-only Transformer 自回归语言模型。

    主干数据流::

        token ids [B,T]
          -> token embedding [B,T,C]
          +  position embedding [T,C]（沿 batch 广播）
          -> L 个 TransformerBlock
          -> final LayerNorm
          -> vocabulary projection
          -> logits [B,T,V]

    训练时，``forward(input_ids, targets)`` 并行处理完整序列；causal mask
    保证位置 t 看不到未来。位置 t 的 logits 用来预测下一个 token。标签
    的右移由数据管线完成，本模型不会在内部自动 shift ``targets``。

    推理分成两个阶段：

    1. ``prefill`` 并行处理 prompt，为每层建立 K/V Cache；
    2. ``decode`` 复用 cache，只处理一个新 token，并追加它的 K/V。

    参数:
        vocab_size: 词表大小 V；token id 应在 ``[0,V-1]``。
        max_seq_len: 最大上下文长度，也是绝对位置表大小，默认 32。
        d_model: embedding 和隐藏状态维度 C，默认 128。
        num_heads: 每个 Block 的注意力头数 H，默认 4，必须整除 C。
        num_layers: TransformerBlock 数量 L，默认 4，至少为 1。

    KV Cache 总体结构:
        ``past_key_values`` 是长度为 L 的 tuple/list；第 l 项是该层的
        ``(key_l,value_l)``，两者均为 ``[B,H,P,D]``。所有层必须拥有
        相同 B 和 P，且每层 H/D 必须与其 Attention 配置一致。
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len=32,
        d_model=128,
        num_heads=4,
        num_layers=4,
    ):
        """初始化 embedding、L 个 Block、最终归一化和词表输出层。"""
        super().__init__()

        # 至少需要一个可索引的位置 0。
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")

        # KV Cache 按层组织；这个实现要求模型至少有一个 Transformer 层。
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1 when using KV cache")

        self.max_seq_len = max_seq_len

        # 离散 token id -> C 维稠密向量：[B,T] -> [B,T,C]。
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 可学习的绝对位置 0..max_seq_len-1；不是相对位置编码或 RoPE。
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # ModuleList 会注册每层参数；列表顺序也是前向顺序和 cache 层顺序。
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # Pre-LN Block 堆叠结束后再做一次最终 LayerNorm。
        self.norm = nn.LayerNorm(d_model)

        # [B,T,C] -> [B,T,V]。此处未与 token_embedding.weight 绑定权重。
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _validate_past_key_values(self, past_key_values, batch_size):
        """检查逐层 KV Cache 的结构与 shape，并返回共同历史长度 P。

        参数:
            past_key_values: None，或长度为 L 的 tuple/list。每一项必须是
                ``(key,value)``，两个 Tensor 同形且为 ``[B,H,P,D]``。
            batch_size: 当前输入的 batch size B，用来确认 cache 与输入属于
                相同数量的样本。

        返回:
            所有层共同的历史序列长度 P；没有 cache 时返回 0。

        本函数检查容器、层数、Tensor 类型、四维 shape、B/H/D 以及各层 P
        是否一致。device/dtype 在 Attention 中与当前 K/V 比较；P+T 是否
        超过最大窗口则由 ``_forward_impl`` 检查。
        """
        if past_key_values is None:
            # None 表示从位置 0 开始，没有历史 token。
            return 0

        # 必须是可稳定按层索引的容器；单个 Tensor 不是完整逐层 cache。
        if not isinstance(past_key_values, (tuple, list)):
            raise ValueError("past_key_values must be a tuple or list")

        # 每个模型层必须有且仅有一对 K/V，缺少或多出都会令层语义错位。
        if len(past_key_values) != len(self.blocks):
            raise ValueError(
                "past_key_values must contain one (key, value) pair per layer"
            )

        # 第一层负责确定 P，后续层与它逐一比较。
        past_length = None

        for layer_index, layer_cache in enumerate(past_key_values):
            # 单层 cache 必须严格包含 key 和 value 两项。
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
                raise ValueError(
                    f"Cache for layer {layer_index} must be a (key, value) pair"
                )

            key, value = layer_cache

            # 先检查类型，再安全访问 ndim/shape，以给出易懂的错误。
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Cache for layer {layer_index} must contain tensors"
                )

            # 四个轴固定为 [batch, heads, cached_sequence, head_dim]。
            # K/V 必须完全同形，才能保证每个 key 都有一一对应的 value。
            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise ValueError(
                    f"Cache for layer {layer_index} must have matching "
                    "[batch, heads, sequence, head_dim] shapes"
                )

            # 读取对应层自身的 H/D，与真正执行注意力的配置保持一致。
            attention = self.blocks[layer_index].attention
            expected_prefix = (batch_size, attention.num_heads)

            # 这里检查 B、H、D；可变的 P 在下方进行跨层一致性检查。
            if key.shape[:2] != expected_prefix or key.size(3) != attention.head_dim:
                raise ValueError(
                    f"Cache for layer {layer_index} has an incompatible shape"
                )

            if past_length is None:
                # 用第一层的 sequence 维建立公共历史长度 P。
                past_length = key.size(2)
            elif key.size(2) != past_length:
                # 每层必须表示同一批历史 token，缓存长度不能各不相同。
                raise ValueError("All layer caches must have the same sequence length")

        # 模型至少有一层，通常 past_length 已赋值；None 分支让返回值更稳健。
        return 0 if past_length is None else past_length

    def _forward_impl(self, input_ids, past_key_values=None, use_cache=False):
        """训练、Prefill、Decode 共用的底层前向实现。

        参数:
            input_ids: 本次要处理的 token id，形状 ``[B,T]``。Embedding
                还要求整数 dtype，且每个 id 位于词表范围内。
            past_key_values: 可选的逐层历史 cache；None 表示 P=0，否则
                是 L 组形状 ``[B,H,P,D]`` 的 K/V。
            use_cache: 是否收集并返回追加当前 token 后的新 cache。

        返回:
            ``(logits, next_cache)``。logits 为 ``[B,T,V]``，只对应本次
            输入的 T 个 token；use_cache=True 时 next_cache 是长度 L 的
            tuple，每层 K/V 为 ``[B,H,P+T,D]``，否则 next_cache 为 None。

        新 token 使用绝对位置 ``P, P+1, ..., P+T-1``，从而让缓存增量
        计算与一次性完整前向的 position embedding 和 logits 对齐。
        """
        # token ids 必须明确区分 batch 与 sequence 两个轴；不接受 [T]。
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        B, T = input_ids.shape

        # 空序列没有可输出的 logits，也无法建立有意义的 Prefill cache。
        if T == 0:
            raise ValueError("input_ids must contain at least one token")

        # 校验每层 cache，并取得它们共同的历史长度 P。
        past_length = self._validate_past_key_values(
            past_key_values,
            batch_size=B,
        )
        total_length = past_length + T

        # 绝对位置表只有 0..max_seq_len-1；这同时限定上下文窗口大小。
        if total_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {total_length} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # [B,T] -> [B,T,C]；相同 token id 先得到相同的 token 向量。
        token_emb = self.token_embedding(input_ids)

        # 无 cache 时从 0 编号；有 P 个历史 token 时从 P 编号。
        # arange 不包含停止端 total_length，恰好生成 T 个位置。
        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device
        )

        # position_emb 为 [T,C]，加到 [B,T,C] 时会沿 batch 轴广播。
        # 因此一个 batch 内所有样本共享同一套位置编号和相同历史长度 P。
        position_emb = self.position_embedding(positions)

        # 把“是什么 token”和“位于哪里”两种信息相加，shape 为 [B,T,C]。
        x = token_emb + position_emb

        # 仅 use_cache=True 时逐层收集 present K/V，最后转换成 tuple。
        present_key_values = []

        for layer_index, block in enumerate(self.blocks):
            # cache 第 l 项只交给第 l 个 Block；不同层的 K/V 不能混用。
            layer_past = (
                None
                if past_key_values is None
                else past_key_values[layer_index]
            )

            if use_cache:
                # x 只含当前 T 个 token 的 hidden states；历史 hidden states
                # 不再重算，而是通过该层历史 K/V 参与注意力。
                x, present_key_value = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                # 保持与 self.blocks 完全相同的层顺序。
                present_key_values.append(present_key_value)
            else:
                # 常规训练不保留 cache，减少返回数据与额外显存占用。
                x = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=False,
                )

        # 最终 LayerNorm 不改变 [B,T,C] shape。
        x = self.norm(x)

        # 每个位置独立投影到词表：[B,T,C] -> [B,T,V]。
        # logits 是未归一化分数；损失或采样阶段再执行 softmax/log-softmax。
        logits = self.lm_head(x)

        # 对外始终返回二元组；关闭 cache 时返回 None，而非空 tuple。
        next_cache = tuple(present_key_values) if use_cache else None
        return logits, next_cache

    def forward(self, input_ids, targets=None):
        """执行常规训练/评估前向，不读取也不返回 KV Cache。

        参数:
            input_ids: token ids，形状 ``[B,T]``。
            targets: 可选监督标签，通常形状 ``[B,T]``。next-token 训练中，
                数据管线应构造 ``targets[:,t] = input_ids[:,t+1]``；这里
                不会自动右移标签。

        返回:
            ``(logits, loss)``。logits 为 ``[B,T,V]``；targets 为 None 时
            loss 也是 None，否则 loss 是 B*T 个位置交叉熵的平均标量。

        若 y[b,t] 是目标 token，则损失为::

            loss = -(1/(B*T)) * sum_{b,t}
                   log softmax(logits[b,t])[y[b,t]]

        ``F.cross_entropy`` 已包含 log-softmax，不能先手动 softmax。
        本方法保留 autograd 计算图，可正常调用 ``loss.backward()``。
        """
        # 常规 forward 从位置 0 处理完整输入，不构建 KV Cache。
        logits, _ = self._forward_impl(input_ids)

        # 即使没有 targets，也固定返回 (logits, None)，保持接口一致。
        loss = None

        if targets is not None:
            B, T, V = logits.shape

            # cross_entropy 需要类别分数 [N,V] 和标签 [N]；这里令 N=B*T，
            # 将 batch/sequence 两轴展平，词表维 V 保持不变。
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def prefill(self, input_ids):
        """一次并行处理完整 prompt，并建立每一层的 KV Cache。

        参数:
            input_ids: prompt token ids，形状 ``[B,T]``，要求
                ``1 <= T <= max_seq_len``。当前模型没有 padding mask，
                所以批量 Prefill 应使用等长且不含 padding 的有效序列。

        返回:
            ``(logits, kv_cache)``：

            - logits 为 ``[B,T,V]``，包含 prompt 每个位置的预测分数；生成
              第一个新 token 时通常只读取 ``logits[:,-1,:]``。
            - kv_cache 是长度 L 的 tuple，每层为 ``(key,value)``，两个
              Tensor 均为 ``[B,H,T,D]``。

        Prefill 仍应用 causal mask，因此 prompt 中位置 t 只能读取 0..t；
        但全部位置可通过矩阵运算并行计算。它为 prompt 做一次完整前向，
        并把每层 K/V 留给后续 Decode，避免未来重复计算 prompt。

        ``torch.no_grad`` 关闭梯度图以节省推理显存和计算开销，但不会自动
        调用 ``model.eval()``；一般仍应由调用方在推理前切换 eval 模式。
        本精简模型当前没有 dropout，所以 train/eval 的数值暂时相同。
        """
        return self._forward_impl(input_ids, use_cache=True)

    @torch.no_grad()
    def decode(self, input_ids, past_key_values):
        """利用已有 KV Cache 增量处理恰好一个新 token。

        参数:
            input_ids: 新 token id，必须为 ``[B,1]``。它通常是根据上次
                Prefill/Decode 的最后一个 logits 采样或 argmax 得到的。
            past_key_values: 上一次 ``prefill`` 或 ``decode`` 返回的逐层
                cache。若历史长度为 P，则必须满足 ``P+1 <= max_seq_len``。

        返回:
            ``(logits, next_cache)``。logits 为 ``[B,1,V]``，用于预测再
            下一个 token；next_cache 含 L 层，每层 K/V 从
            ``[B,H,P,D]`` 增长为 ``[B,H,P+1,D]``。

        历史 token 的 K/V 已经保存，本次只为新 token 计算 Q/K/V。新 query
        与 P+1 个 key 做注意力，不必让全部历史 token 再经过所有投影和
        Block。cache 会逐 token 增长；到达 ``max_seq_len`` 后，底层校验
        会拒绝继续追加。项目 ``generate.py`` 此时会截取最近窗口重新
        Prefill，从而重置可学习绝对位置和 cache。

        时序上，上一次 logits 先预测出这个新 token；将该 token 传入
        Decode 后，本次返回的 logits 才用于预测它后面的 token。新 cache
        由拼接得到，不会原地修改调用方持有的旧 cache。
        """
        # 公开 Decode 接口故意限制为单 token，避免和多 token Prefill 混淆。
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError("decode expects input_ids with shape [batch, 1]")

        # 无历史上下文时应调用 prefill；decode 必须从已有 cache 继续。
        if past_key_values is None:
            raise ValueError("decode requires past_key_values from prefill or decode")

        return self._forward_impl(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
