"""一个最小化的、按空白切词的词级分词器。

这个项目的重点是学习 Transformer，因此这里没有使用 BPE、WordPiece、
SentencePiece 等工业级子词算法，而是采用最容易看懂的实现：

1. 把语料统一转换为小写；
2. 按空白字符切成单词；
3. 用训练语料中出现过的单词构建词表；
4. 在“单词”和“整数 token id”之间互相转换。

需要特别注意：

* 这是“词级” tokenizer，不是字符级 tokenizer。
* ``str.split()`` 只按空白切分，不会单独处理标点。例如 ``"hello,"``
  和 ``"hello"`` 会被视为两个不同 token。
* 对没有空格的中文文本，一整段文字可能会被当作一个 token。因此若要用中文
  语料训练，通常需要预先分词，或把本实现替换成字符级/子词级 tokenizer。
* 词表完全由传入的 ``text`` 决定；训练和推理必须使用同一份词表映射。
"""


class SimpleTokenizer:
    """在给定语料上构建的简单词级分词器。

    参数
    ----
    text:
        用来构建词表的完整训练语料。构造函数只分析文本并建立映射，不会保存
        原始文本本身。

    主要属性
    --------
    itos:
        ``index to string`` 的缩写。它是一个列表，位置（下标）就是 token id，
        对应元素是 token 字符串，即 ``itos[token_id] -> token``。
    stoi:
        ``string to index`` 的缩写。它是 ``itos`` 的反向字典，即
        ``stoi[token] -> token_id``。
    pad_id:
        ``<pad>`` 的整数 id。当前训练脚本把所有样本都截成相同长度，所以没有
        真正插入 padding；保留它是为了以后支持变长 batch。
    unk_id:
        ``<unk>`` 的整数 id。推理文本中遇到词表外单词时，用它代替未知单词。

    这里没有 ``<bos>``（句子开始）或 ``<eos>``（句子结束）token，因此模型
    学到的是训练文件中连续 token 的 next-token 关系，而不是显式的句子边界。
    """

    def __init__(self, text):
        """根据 ``text`` 构建固定词表以及双向映射。"""

        # lower() 让英文大小写共享同一个 token，例如 "The" 和 "the" 都会
        # 变成 "the"。这可以缩小词表，但也会永久丢失原始大小写信息。
        #
        # split() 不传分隔符时，会按任意连续空白（空格、换行、制表符等）
        # 切分，并自动忽略开头、结尾及连续空白产生的空字符串。
        words = text.lower().split()

        # set(words) 去重，只保留每种单词的一份；sorted(...) 再排序，使相同
        # 语料每次构造出的 token id 顺序确定，不受集合内部顺序影响。
        # 注意：这里只保留“有哪些词”，没有保存词频。
        vocab = sorted(set(words))

        # 在普通词表前手动加入两个特殊 token：
        #   id=0 -> <pad>：未来可用于把不同长度的序列补齐；
        #   id=1 -> <unk>：表示未登录词（out-of-vocabulary word）。
        # 所有来自语料的普通单词 id 都从 2 开始。
        #
        # itos 的列表下标天然就是模型使用的整数 token id。
        self.itos = ["<pad>", "<unk>"] + vocab

        # enumerate(self.itos) 依次产生 (idx, word)，字典推导式将其反转为
        # word -> idx。由于 itos 中没有重复项，这是一一对应的映射。
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

        # 不直接写死 0 和 1，而是从 stoi 查询特殊 token 的 id。这样即使以后
        # 调整特殊 token 的排列顺序，下面的 encode 逻辑仍然能够正确工作。
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self):
        """返回词表大小（两个特殊 token 也计算在内）。

        使用 ``@property`` 后，调用方可以写 ``tokenizer.vocab_size``，看起来
        像读取普通属性；实际值则始终由当前 ``itos`` 的长度动态计算。

        模型的 token embedding 输入维度和最终语言模型头输出维度都需要这个值。
        """

        return len(self.itos)

    def encode(self, text):
        """把文本编码为 token id 列表。

        编码过程必须与构建词表时的预处理完全一致：先转小写，再按空白切词。
        对每个单词：

        * 若它存在于 ``stoi``，返回对应 id；
        * 若它从未出现在建表语料中，返回 ``unk_id``。

        返回的是普通 Python ``list[int]``。训练脚本稍后会把它转换成
        ``torch.long`` 张量，供 embedding 层和交叉熵损失使用。

        ``<pad>`` 不会在这里被自动添加；本方法也不会自动截断或补齐序列。
        """

        return [
            # dict.get(key, default) 在 key 不存在时返回 unk_id，而不会抛出
            # KeyError。这就是推理阶段处理词表外单词的机制。
            self.stoi.get(word, self.unk_id)
            for word in text.lower().split()
        ]

    def decode(self, ids):
        """把一串 token id 解码为由空格连接的文本。

        ``self.itos[i]`` 完成 id -> token 的反向查询，``" ".join(...)`` 再
        用单个空格连接所有 token。因为编码时已经小写化且丢弃了原始空白，
        decode 只能还原“规范化后的 token 序列”，不能逐字符恢复原文格式。

        本方法会原样显示 ``<pad>`` 和 ``<unk>``，也不会过滤特殊 token。
        如果传入超出词表范围的 id，列表索引会抛出 ``IndexError``；这是为了
        让无效模型输出或错误词表尽早暴露。
        """

        return " ".join(self.itos[i] for i in ids)
