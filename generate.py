import torch

from model import TransformerLanguageModel


# 这里的 device 只服务于“默认运行时”加载：
# - 有 CUDA 时，把 checkpoint 中的张量和新建模型都放到 GPU；
# - 否则使用 CPU。
#
# generate() 还支持由调用者传入 model/stoi/itos（依赖注入）。在这种情况下，真正
# 使用哪个设备由传入模型的参数决定，而不是强制使用这个全局 device。这样测试可以
# 注入一个很小的 CPU 假模型，实际应用也可以注入已经放到指定 GPU 上的模型。
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 默认运行时缓存。其最终值是 (model, stoi, itos)，但在第一次真正需要生成、编码或
# 解码之前保持为 None。这是一种“懒加载（lazy loading）”：仅仅 import generate.py
# 不会立刻读取 model.pt，也不会立刻构造模型或占用 GPU 显存。
#
# 这对单元测试尤其重要：测试通过依赖注入传入 RecordingModel 和词表，因此完全不需要
# 本地存在 model.pt。第一次调用 _get_default_runtime() 后，结果会保存在这里，后续调用
# 复用同一个模型和词表，避免重复加载 checkpoint。
_default_runtime = None


def load_runtime(checkpoint_path="model.pt"):
    """从 checkpoint 构造默认的推理模型与双向词表。

    checkpoint 需要至少包含三个键：

    - ``model_state_dict``：模型参数；
    - ``stoi``（string to index）：``单词 -> token id`` 的字典；
    - ``itos``（index to string）：``token id -> 单词`` 的序列。

    ``map_location=device`` 保证即使 checkpoint 原本是在另一种设备上保存的，也会在
    当前可用设备上恢复。模型结构参数必须与训练并保存该 checkpoint 时一致，否则
    load_state_dict() 会报告权重形状不匹配。
    """
    # 只在该函数被调用时才访问磁盘；这正是上面“懒加载”的实际加载入口。
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # itos 和 stoi 必须来自同一次训练/同一个 tokenizer。若两者映射不一致，模型输出的
    # token id 会被翻译成错误的单词，输入单词也会被送入错误的 embedding 行。
    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]

    # 这里重建的网络超参数要和 checkpoint 对应。len(itos) 就是词表大小：合法 token
    # id 的范围为 [0, len(itos) - 1]。
    model = TransformerLanguageModel(
        vocab_size=len(itos),
        max_seq_len=8,
        d_model=128,
        num_heads=4,
        num_layers=3,
    ).to(device)

    # state_dict 只保存参数/缓冲区，不保存 Python 模型对象本身；因此先按相同结构实例化，
    # 再把训练好的权重装入。
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    # 切到推理模式。当前这个迷你模型没有 Dropout/BatchNorm，但显式调用 eval() 仍是正确
    # 的推理契约，也使以后加入这类层时不会因忘记切换模式而产生不稳定结果。
    model.eval()

    return model, stoi, itos


def _get_default_runtime():
    """按需创建并缓存默认 ``(model, stoi, itos)``。"""
    global _default_runtime

    # 第一次调用才加载；之后直接返回同一组对象。这既减少磁盘 I/O，也避免反复分配模型
    # 参数与 GPU 显存。该示例按单进程、顺序调用设计，没有额外处理多线程首次加载竞争。
    if _default_runtime is None:
        _default_runtime = load_runtime()

    return _default_runtime


def encode(text, stoi=None):
    """把文本转换成 token id 列表。

    这是一个最小的“按空白分词”的 tokenizer：先转小写，再用 split() 按任意空白切分。
    它不会处理子词、标点拆分或 Unicode 规范化，所以其规则必须与训练时保持一致。

    参数 ``stoi`` 可由调用者注入；省略时才触发默认 checkpoint 的懒加载。词表中没有的
    单词统一映射到 ``<unk>`` 的 id，因此 stoi 必须包含 ``"<unk>"``。
    """
    if stoi is None:
        _, stoi, _ = _get_default_runtime()

    # dict.get(word, unknown_id) 实现 OOV（out-of-vocabulary）回退。
    return [
        stoi.get(word, stoi["<unk>"])
        for word in text.lower().split()
    ]


def decode(ids, itos=None):
    """把 token id 序列映射回以空格连接的文本。

    ``itos[i]`` 与 encode() 中的 ``stoi[token]`` 应互为逆映射。这里不执行越界容错：模型
    的输出维度应等于词表大小，因此正常采样得到的 id 天然位于合法范围内。
    """
    if itos is None:
        _, _, itos = _get_default_runtime()

    return " ".join(itos[i] for i in ids)


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=20,
    *,
    model=None,
    stoi=None,
    itos=None,
):
    """使用 Prefill + KV Cache 自回归生成文本。

    Args:
        prompt: 输入提示文本，使用 encode() 的小写、空白分词规则。
        max_new_tokens: 最多追加多少个新 token；不包含 prompt 自身的 token 数。
        model, stoi, itos: 可选的依赖注入参数。三者必须一起提供或一起省略。一起省略时
            使用懒加载的默认 checkpoint；一起提供时，generate() 不读取 model.pt。

    核心时序如下：

    1. ``prefill(prompt)`` 一次性处理提示词并建立每层 KV cache；
    2. prompt 最后一个位置的 logits 已经表示“下一个 token”的分布，可直接采样第 1 个
       新 token，不需要先调用 decode()；
    3. 若还需要继续生成，``decode(刚采样的 token, cache)`` 消费这一个 token，返回用来
       预测再下一个 token 的 logits，并把该 token 的 K/V 追加到 cache；
    4. cache 达到 max_seq_len 后，用最新滑动窗口重新 prefill。

    注意第 2、3 步带来的 off-by-one：生成 N 个新 token 时，需要 1 次 prefill，但只需
    N-1 次“decode 或 re-prefill”更新；最后一个 token 采样完成后已经得到最终结果，没有
    必要再把它送入模型。这也是循环中提前 break 的原因。
    """
    # “生成负数个 token”没有定义，尽早给出比底层循环更清楚的错误信息。
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    runtime_values = (model, stoi, itos)

    # 三个值全为 None：使用默认运行时，并在这里按需触发 checkpoint 懒加载。
    if all(value is None for value in runtime_values):
        model, stoi, itos = _get_default_runtime()
    # 三个依赖必须作为一个完整、彼此匹配的运行时一起注入。只替换其中一部分可能导致
    # 模型词表维度与映射不一致，所以明确拒绝“半注入”状态。
    elif any(value is None for value in runtime_values):
        raise ValueError("model, stoi, and itos must be provided together")

    # ids 是 prompt 的完整 token 序列，后面会保留完整历史用于最终返回以及满窗时重建
    # cache。真正送入模型的上下文可能会因 max_seq_len 限制而截取为最新窗口。
    ids = encode(prompt, stoi)

    # 不使用全局 device，而从模型参数读取实际设备。这样传入的 CPU/GPU 模型都能正确
    # 工作，也避免 input_ids 与模型权重不在同一设备导致运行时报错。
    model_device = next(model.parameters()).device

    # 模型输入约定为 [batch, sequence]。这个生成函数一次只处理一个 prompt，所以 batch=1；
    # token id 是 embedding 的索引，dtype 必须是 torch.long。
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=model_device,
    )

    # 0 个新 token 时直接把 prompt 解码返回，完全不调用模型。这也允许空 prompt 在该特例
    # 下返回空字符串；顺序不能随意挪到下面的空 prompt 检查之后。
    if max_new_tokens == 0:
        return decode(x[0].tolist(), itos)

    # prefill 至少需要一个 token 才能从“最后一个位置”的 logits 预测后续 token。
    if not ids:
        raise ValueError("prompt must contain at least one token")

    # 模型的位置 embedding 只定义了 [0, max_seq_len - 1]，因此过长 prompt 只保留最新的
    # max_seq_len 个 token。x 仍保存完整 prompt；x_context 才是本轮模型可见的滑动窗口。
    x_context = x[:, -model.max_seq_len:]

    # Prefill 一次并行处理整个可见 prompt：
    # - logits: [1, context_length, vocab_size]；最后一个位置可预测第 1 个新 token；
    # - kv_cache: 真实模型中是“每层一个 (K, V)”的元组，保存所有上下文 token 的 K/V。
    logits, kv_cache = model.prefill(x_context)

    # 显式记录 cache 当前覆盖的 token 数。generate() 把 cache 当作模型私有的、不透明对象，
    # 不依赖其内部张量布局；这也让 RecordingModel 可以用一个整数模拟 cache。
    cached_length = x_context.size(1)

    for step in range(max_new_tokens):
        # 无论 logits 来自初次 prefill 还是上一次 decode/re-prefill，其最后一个序列位置
        # 都对应当前上下文之后“下一个 token”的未归一化分数，形状为 [batch=1, vocab]。
        next_token_logits = logits[:, -1, :]

        # softmax 把 logits 归一化为类别概率。本示例没有 temperature、top-k、top-p 或重复
        # 惩罚，因此直接从模型的完整词表分布采样；结果通常具有随机性。
        probs = torch.softmax(next_token_logits, dim=-1)

        # multinomial 按概率分布抽取一个 token，返回形状 [1, 1]，恰好符合 decode() 要求。
        # 若要贪心生成，可改成 argmax；但那会改变当前程序的生成策略与测试之外的行为。
        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        # x 保存“完整 prompt + 已生成 token”。即使模型每步只消费一个新 token，也不能丢掉
        # 这份完整序列，因为最终需要返回全部文本，cache 满时还需要从其尾部重建窗口。
        x = torch.cat(
            [x, next_token],
            dim=1,
        )

        # 已采样最后一个所需 token 时立即结束。此时无需 decode(next_token)：decode 的输出
        # 只会用于预测第 max_new_tokens+1 个 token，而调用者并未请求它。于是生成 N 个 token
        # 时，cache 更新恰好发生 N-1 次，而不是 N 次（典型的 off-by-one 易错点）。
        if step == max_new_tokens - 1:
            break

        if cached_length < model.max_seq_len:
            # cache 尚有空间：只计算刚生成 token 的 Q/K/V。历史 token 的 K/V 已在 cache 中，
            # 无需像普通 forward 那样重复计算整个前缀。返回的 logits 形状为 [1, 1, vocab]，
            # 这个唯一位置会在下一轮用于采样。
            logits, kv_cache = model.decode(next_token, kv_cache)

            # decode 已把一个 token 追加到每层 cache，所以长度增加 1。注意该变量描述的是
            # “已被模型消费并进入 cache 的 token 数”，不是 x 的总长度；刚采样但尚未 decode
            # 的 token 要到上面的调用完成后才计入。
            cached_length += 1
        else:
            # cache 已达到 max_seq_len，不能继续追加，否则位置 embedding 会越界，模型自身也会
            # 拒绝 total_length > max_seq_len。这里保留 x 的最新窗口，并重新执行 prefill。
            #
            # 为什么不能只删掉 cache 中最老的一列 K/V？该模型使用“学习得到的绝对位置
            # embedding”。滑窗后，原先位于位置 1 的 token 应改为新窗口的位置 0；但缓存的
            # K/V 已混合了旧的位置 embedding，简单切片无法把它改成位置 0 的表示。同时新 token
            # 也需要使用窗口内最后一个合法位置。因此必须对整个最新窗口重新 prefill，让位置
            # 编号重置为 0..max_seq_len-1，并得到语义一致的新 cache。
            x_context = x[:, -model.max_seq_len:]
            logits, kv_cache = model.prefill(x_context)
            cached_length = x_context.size(1)

    # .tolist() 把设备上的 LongTensor 搬回普通 Python id 列表，再由 itos 映射成人类可读文本。
    return decode(x[0].tolist(), itos)


if __name__ == "__main__":
    # 直接运行本文件时才执行演示；被测试或其他模块 import 时不会自动生成文本。
    print(generate("the cat", max_new_tokens=15))
    print(generate("the model", max_new_tokens=15))
    print(generate("attention", max_new_tokens=15))
