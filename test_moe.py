"""验证 MoE 的分发、训练梯度、缓存推理和 checkpoint 加载。"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import generate, load_runtime
from model import MoE, TransformerLanguageModel


class MoETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_routes_only_selected_tokens_and_combines_experts(self):
        moe = MoE(d_model=2, num_experts=4, top_k=2)
        # 用固定输出的专家，手算三个 token 的加权结果；专家 3 不应被调用。
        values = torch.tensor([[1., 0.], [0., 2.], [-1., -1.], [10., 10.]])
        moe.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        seen = {}
        with torch.no_grad():
            moe.router.weight.copy_(torch.tensor([[3., 0.], [0., 2.], [-2., -1.], [0., 0.]]))
            moe.router.bias.copy_(torch.tensor([0., 0., 0., -10.]))
            for expert_id, expert in enumerate(moe.experts):
                expert.weight.zero_()
                expert.bias.copy_(values[expert_id])
                expert.register_forward_pre_hook(
                    lambda module, args, eid=expert_id: seen.__setitem__(eid, args[0].detach().clone())
                )

        x = torch.tensor([[[1., 0.], [0., 1.], [-1., -1.]]])
        output, aux_loss = moe(x)
        w01 = torch.tensor([3., 0.]).softmax(dim=0)
        w10 = torch.tensor([2., 0.]).softmax(dim=0)
        w21 = torch.tensor([3., -2.]).softmax(dim=0)
        expected = torch.stack([
            w01[0] * values[0] + w01[1] * values[1],
            w10[0] * values[1] + w10[1] * values[0],
            w21[0] * values[2] + w21[1] * values[1],
        ]).unsqueeze(0)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(seen[0], x[0, :2])
        torch.testing.assert_close(seen[1], x[0])
        torch.testing.assert_close(seen[2], x[0, 2:])
        self.assertNotIn(3, seen)

        # 六次路由中各专家分别收到 2、3、1、0 次；完整概率参与辅助损失。
        probs = torch.tensor([[3., 0., -2., -10.], [0., 2., -1., -10.], [-3., -2., 3., -10.]]).softmax(dim=-1)
        expected_aux = 4 * (torch.tensor([2., 3., 1., 0.]) / 6 * probs.mean(dim=0)).sum()
        torch.testing.assert_close(aux_loss, expected_aux)

    def test_training_loss_and_router_gradients(self):
        model = TransformerLanguageModel(
            vocab_size=16, max_seq_len=8, d_model=8, num_heads=2,
            num_layers=2, num_experts=4, top_k=2,
        )
        # 固定每层路由分布，便于手算辅助损失，并确保有两个专家未被选择。
        with torch.no_grad():
            for block in model.blocks:
                block.moe.router.weight.zero_()
                block.moe.router.bias.copy_(torch.tensor([2., 1., 0., -1.]))
        x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
        logits, loss = model(x, targets)
        ce_loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
        probs = torch.tensor([2., 1., 0., -1.]).softmax(dim=0)
        expected_aux = 4 * (probs[0] + probs[1]) / 2
        torch.testing.assert_close(loss, ce_loss + 0.01 * expected_aux)
        loss.backward()
        for block in model.blocks:
            self.assertGreater(block.moe.router.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.moe.experts[0].net[0].weight.grad.abs().sum().item(), 0)
            self.assertIsNone(block.moe.experts[2].net[0].weight.grad)

        # 关闭辅助损失后，top-2 的加权路径本身仍能训练 router。
        model.zero_grad(set_to_none=True)
        model.aux_loss_coef = 0
        _, main_loss = model(x, targets)
        main_loss.backward()
        self.assertGreater(model.blocks[0].moe.router.weight.grad.abs().sum().item(), 0)

    def test_top_one_router_learns_from_auxiliary_loss(self):
        moe = MoE(d_model=2, num_experts=3, top_k=1)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.bias.copy_(torch.tensor([2., 0., -1.]))
        output, aux_loss = moe(torch.ones(1, 2, 2))
        self.assertEqual(output.shape, (1, 2, 2))
        aux_loss.backward()
        self.assertGreater(moe.router.weight.grad.abs().sum().item(), 0)

    def test_cached_decode_matches_full_forward(self):
        model = TransformerLanguageModel(
            vocab_size=16, max_seq_len=8, d_model=8, num_heads=2,
            num_layers=2,
        ).eval()
        x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        expected, loss = model(x)
        self.assertIsNone(loss)
        logits, cache = model.prefill(x[:, :2])
        torch.testing.assert_close(logits, expected[:, :2], rtol=1e-5, atol=1e-6)
        for position in range(2, x.size(1)):
            logits, cache = model.decode(x[:, position:position + 1], cache)
            torch.testing.assert_close(logits, expected[:, position:position + 1], rtol=1e-5, atol=1e-6)
        for key, value in cache:
            self.assertEqual(key.shape, (2, 2, 5, 4))
            self.assertEqual(value.shape, key.shape)

    def test_checkpoint_round_trip_and_generation(self):
        itos = ["<pad>", "<unk>", "a", "b"]
        stoi = {word: i for i, word in enumerate(itos)}
        config = dict(
            max_seq_len=4, d_model=8, num_heads=2, num_layers=1,
            num_experts=3, top_k=2, aux_loss_coef=0.02,
        )
        model = TransformerLanguageModel(vocab_size=len(itos), **config).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "moe.pt"
            torch.save(dict(model_state_dict=model.state_dict(), model_config=config, stoi=stoi, itos=itos), path)
            restored, restored_stoi, restored_itos = load_runtime(path)
        self.assertIsInstance(restored.blocks[0].moe, MoE)
        self.assertEqual(restored.aux_loss_coef, 0.02)
        x = torch.tensor([[2, 3]])
        expected, _ = model(x)
        actual, _ = restored(x.to(next(restored.parameters()).device))
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
        # 超过窗口，覆盖一次 MoE decode 和满窗后的重新 prefill。
        result = generate("a b", max_new_tokens=5, model=restored, stoi=restored_stoi, itos=restored_itos)
        self.assertEqual(len(result.split()), 7)


if __name__ == "__main__":
    unittest.main()
