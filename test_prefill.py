import unittest

import torch

from generate import generate
from model import TransformerLanguageModel


class RecordingModel(torch.nn.Module):
    """用于精确观察 generate() 调用时序的轻量假模型。

    这里不使用真正的 Transformer，是因为生成流程测试关注的是“何时 prefill、何时 decode、
    每次传入了哪些 token”，而不是神经网络数值本身。假模型会记录所有调用，并始终让 token
    0 拥有 100% 的采样概率，因此测试既快速又确定，不依赖训练好的 model.pt。

    它只实现 generate() 所依赖的最小接口：max_seq_len、parameters()、prefill() 和 decode()。
    这也演示了 generate() 的依赖注入设计：任何满足该协议的模型都可以被传入。
    """

    def __init__(self, max_seq_len=4, vocab_size=7):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # generate() 会用 next(model.parameters()).device 决定输入张量的设备。这个参数不参与
        # 计算，只是让假模型像真实 nn.Module 一样拥有可查询设备的 parameter。
        self.anchor = torch.nn.Parameter(torch.zeros(1))

        # 每个元素保存 ("prefill"/"decode", 输入快照)，供断言调用顺序和 token 内容。
        self.calls = []

    def _next_token_logits(self, input_ids):
        """构造一个确定性分布：任意位置都只能采样到 token 0。"""
        # 输出形状与真实模型一致：[batch, sequence, vocab_size]。先把所有 token 的 logit
        # 设成 -inf，再把 token 0 设成 0；softmax 后 token 0 概率为 1，其余概率为 0。
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            float("-inf"),
            device=input_ids.device,
        )
        logits[..., 0] = 0
        return logits

    def prefill(self, input_ids):
        """记录完整上下文，并用上下文长度这个整数模拟 KV cache。"""
        # detach() 避免保留计算图，cpu() 让断言与测试中的 CPU tensor 可直接比较，clone()
        # 确保后续即使原 tensor 被修改，记录仍是调用当时不可变的快照。
        self.calls.append(("prefill", input_ids.detach().cpu().clone()))

        # 真实模型返回逐层 (K, V)；这里返回一个整数长度即可，因为 generate() 不检查 cache
        # 的内部格式，只会原样把它交回 decode()。这正体现了 cache 对调用者是不透明对象。
        return self._next_token_logits(input_ids), input_ids.size(1)

    def decode(self, input_ids, kv_cache):
        """记录单 token decode，并模拟 cache 长度增加 1。"""
        self.calls.append(("decode", input_ids.detach().cpu().clone()))
        return self._next_token_logits(input_ids), kv_cache + 1


class PrefillTest(unittest.TestCase):
    """验证真实 Transformer 的 Prefill/KV Cache 数值正确性与接口边界。"""

    def setUp(self):
        """为每个测试创建同一配置、同一随机初始化的小模型。"""
        # 固定随机种子，使模型初始化可复现。每个 test 前 unittest 都会重新执行 setUp，
        # 因而测试之间不会共享 cache、梯度或其他可变状态。
        torch.manual_seed(0)
        self.model = TransformerLanguageModel(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            num_heads=4,
            num_layers=2,
        )
        # 明确采用推理模式；当前模型无 Dropout，但这与生产推理调用保持一致。
        self.model.eval()

    def test_prefill_matches_regular_forward(self):
        """Prefill 的 logits 应与无缓存完整 forward 完全一致，并建立正确形状的逐层 cache。"""
        # B=2、T=4。两个 batch 样本同时验证 cache 的 batch 维没有被误处理。
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        # 普通 forward 是数值基准；prefill 应只多返回 cache，不应改变 logits。
        expected_logits, _ = self.model(input_ids)
        prefill_logits, kv_cache = self.model.prefill(input_ids)

        torch.testing.assert_close(prefill_logits, expected_logits)
        # num_layers=2，所以 cache 中应恰好有 2 组 (key, value)，每层一组。
        self.assertEqual(len(kv_cache), 2)

        for key, value in kv_cache:
            # cache 布局为 [batch, heads, sequence, head_dim]：
            # batch=2，num_heads=4，sequence=4，head_dim=d_model/num_heads=16/4=4。
            self.assertEqual(key.shape, (2, 4, 4, 4))
            self.assertEqual(value.shape, (2, 4, 4, 4))

    def test_cached_decode_matches_regular_forward(self):
        """逐 token 缓存解码的每一步 logits 应匹配一次性完整 forward 的对应位置。"""
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ])
        # 完整 forward 同时计算所有 6 个位置，作为可信的数值基准。
        expected_logits, _ = self.model(input_ids)

        # 先把前 3 个 token 放进 cache；prefill 的 logits 在本测试中无需使用。
        _, kv_cache = self.model.prefill(input_ids[:, :3])

        # 再依次“消费”位置 3、4、5 的 token。decode 返回的是输入这个位置本身对应的
        # logits；它应等于完整 forward 在同一绝对位置的结果。这同时验证了：
        # 1) 历史 K/V 拼接顺序正确；2) causal mask 正确；3) 位置编号从 past_length 继续。
        for position in range(3, input_ids.size(1)):
            step_logits, kv_cache = self.model.decode(
                input_ids[:, position:position + 1],
                kv_cache,
            )
            torch.testing.assert_close(
                step_logits,
                expected_logits[:, position:position + 1],
                # 完整计算与分步矩阵计算可能因浮点运算顺序产生极小误差，因此使用合理容差，
                # 而不是要求逐 bit 相等。
                rtol=1e-5,
                atol=1e-6,
            )

        # prefill 3 个，再 decode 3 个，最终每层 cache 都应覆盖完整的 6 个 token。
        for key, value in kv_cache:
            self.assertEqual(key.size(2), input_ids.size(1))
            self.assertEqual(value.size(2), input_ids.size(1))

    def test_training_forward_contract_is_unchanged(self):
        """加入推理 cache API 后，原有训练 forward、loss 与反向传播契约必须保持不变。"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        # 语言模型训练目标通常是输入右移一位后的 token；这里只需验证 loss 路径可用。
        targets = torch.tensor([[2, 3, 4, 5]])

        result = self.model(input_ids, targets)

        # 原接口仍返回 (logits, loss)，而不是因 cache 功能变成三元组。
        self.assertEqual(len(result), 2)
        logits, loss = result
        self.assertEqual(logits.shape, (1, 4, 32))
        self.assertIsNotNone(loss)

        # 能从交叉熵一直反传到 token embedding，说明新增推理路径没有破坏训练计算图。
        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)

    def test_invalid_prefill_and_decode_inputs_are_rejected(self):
        """模型应在非法长度、空输入和错误 decode 形状处尽早给出明确错误。"""
        # 至少需要一个合法位置 embedding。
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            TransformerLanguageModel(vocab_size=8, max_seq_len=0)

        # KV cache 以“每层一个 (K,V)”组织，因而缓存模式至少需要一层。
        with self.assertRaisesRegex(ValueError, "num_layers"):
            TransformerLanguageModel(vocab_size=8, num_layers=0)

        # 空序列没有最后位置，也就无法提供用于预测下一个 token 的 logits。
        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.model.prefill(torch.empty((1, 0), dtype=torch.long))

        # max_seq_len=8 的 cache 已满，再追加一个 token 会使总长度变成 9，必须拒绝。
        full_context = torch.arange(8).unsqueeze(0)
        _, kv_cache = self.model.prefill(full_context)

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            self.model.decode(torch.tensor([[8]]), kv_cache)

        # decode API 刻意限定每次只处理一个新 token，即输入形状必须是 [batch, 1]。
        with self.assertRaisesRegex(ValueError, r"\[batch, 1\]"):
            self.model.decode(torch.tensor([[1, 2]]), kv_cache)


class GenerateTest(unittest.TestCase):
    """使用 RecordingModel 验证生成循环的调用时序、滑窗内容和边界行为。"""

    # 固定的双向词表。token 0（<gen>）是 RecordingModel 唯一会采样到的 token，因而
    # 每个生成结果完全确定；<unk> 则满足 encode() 对未知词回退项的要求。
    stoi = {
        "<gen>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "<unk>": 6,
    }
    itos = ["<gen>", "a", "b", "c", "d", "e", "<unk>"]

    def test_generate_prefills_once_then_decodes_new_tokens(self):
        """cache 有空间时，应 prefill 一次，之后只 decode 新生成的 token。"""
        model = RecordingModel(max_seq_len=4)

        result = generate(
            "a b",
            max_new_tokens=3,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        # 假模型每步只能生成 token 0，即 <gen>。
        self.assertEqual(result, "a b <gen> <gen> <gen>")

        # 生成 3 个 token 的精确时序：
        # - prefill("a b") 的最后 logits 直接采样第 1 个 <gen>；
        # - decode(第 1 个 <gen>) 产生用于采样第 2 个的 logits；
        # - decode(第 2 个 <gen>) 产生用于采样第 3 个的 logits；
        # - 第 3 个已经是最终 token，所以不再 decode。
        # 因此是 1 次 prefill + (3 - 1) 次 decode，这个断言专门防止 off-by-one。
        self.assertEqual([kind for kind, _ in model.calls], [
            "prefill",
            "decode",
            "decode",
        ])

        # prefill 收到完整 prompt；两次 decode 都只收到刚生成的单个 token 0。
        torch.testing.assert_close(model.calls[0][1], torch.tensor([[1, 2]]))
        torch.testing.assert_close(model.calls[1][1], torch.tensor([[0]]))
        torch.testing.assert_close(model.calls[2][1], torch.tensor([[0]]))

    def test_generate_reprefills_latest_window_when_cache_is_full(self):
        """cache 满后应对最新窗口重新 prefill，而不是继续 decode 或错误截断旧 K/V。"""
        model = RecordingModel(max_seq_len=4)

        generate(
            "a b c",
            max_new_tokens=3,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        # 初始 prompt 长 3：prefill 后 cache 长 3。
        # 第 1 个 <gen> 由 prefill logits 采样，再 decode 它，使 cache 长 4（刚好满窗）。
        # 第 2 个 <gen> 由该 decode logits 采样；为了预测第 3 个，不能再向满 cache 追加，
        # 所以对最新 4 个 token 重新 prefill。第 3 个采样后结束，无需再更新 cache。
        self.assertEqual([kind for kind, _ in model.calls], [
            "prefill",
            "decode",
            "prefill",
        ])

        # 完整序列此时为 [a, b, c, <gen>, <gen>]，最新长度 4 的窗口应丢弃 a，得到
        # [b, c, <gen>, <gen>] => ids [2, 3, 0, 0]。
        torch.testing.assert_close(
            model.calls[2][1],
            torch.tensor([[2, 3, 0, 0]]),
        )

    def test_generate_prefills_only_the_latest_prompt_window(self):
        """prompt 本身超长时，第一次 prefill 就应只接收最后 max_seq_len 个 token。"""
        model = RecordingModel(max_seq_len=4)

        generate(
            "a b c d e",
            max_new_tokens=1,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        # 只请求 1 个新 token：其分布已经由 prefill 最后位置提供，所以模型总调用数为 1，
        # 不应在采样完成后多做一次无用 decode。
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], "prefill")

        # prompt ids 为 [1,2,3,4,5]，max_seq_len=4，只保留最新 [2,3,4,5]。
        torch.testing.assert_close(
            model.calls[0][1],
            torch.tensor([[2, 3, 4, 5]]),
        )

    def test_zero_new_tokens_does_not_run_the_model(self):
        """请求生成 0 个 token 时应原样返回 prompt，且不执行 prefill/decode。"""
        model = RecordingModel(max_seq_len=4)

        result = generate(
            "a b",
            max_new_tokens=0,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        self.assertEqual(result, "a b")
        # 空调用列表证明 generate() 在 max_new_tokens=0 的快速路径中没有运行模型。
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    # 允许直接执行 python test_prefill.py；被测试发现器 import 时不会重复启动测试。
    unittest.main()
