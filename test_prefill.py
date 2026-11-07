import unittest

import torch

from generate import generate
from model import TransformerLanguageModel


class RecordingModel(torch.nn.Module):
    def __init__(self, max_seq_len=4, vocab_size=7):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def _next_token_logits(self, input_ids):
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            float("-inf"),
            device=input_ids.device,
        )
        logits[..., 0] = 0
        return logits

    def prefill(self, input_ids):
        self.calls.append(("prefill", input_ids.detach().cpu().clone()))
        return self._next_token_logits(input_ids), input_ids.size(1)

    def decode(self, input_ids, kv_cache):
        self.calls.append(("decode", input_ids.detach().cpu().clone()))
        return self._next_token_logits(input_ids), kv_cache + 1


class PrefillTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TransformerLanguageModel(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            num_heads=4,
            num_layers=2,
        )
        self.model.eval()

    def test_prefill_matches_regular_forward(self):
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        expected_logits, _ = self.model(input_ids)
        prefill_logits, kv_cache = self.model.prefill(input_ids)

        torch.testing.assert_close(prefill_logits, expected_logits)
        self.assertEqual(len(kv_cache), 2)

        for key, value in kv_cache:
            self.assertEqual(key.shape, (2, 4, 4, 4))
            self.assertEqual(value.shape, (2, 4, 4, 4))

    def test_cached_decode_matches_regular_forward(self):
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ])
        expected_logits, _ = self.model(input_ids)

        _, kv_cache = self.model.prefill(input_ids[:, :3])

        for position in range(3, input_ids.size(1)):
            step_logits, kv_cache = self.model.decode(
                input_ids[:, position:position + 1],
                kv_cache,
            )
            torch.testing.assert_close(
                step_logits,
                expected_logits[:, position:position + 1],
                rtol=1e-5,
                atol=1e-6,
            )

        for key, value in kv_cache:
            self.assertEqual(key.size(2), input_ids.size(1))
            self.assertEqual(value.size(2), input_ids.size(1))

    def test_training_forward_contract_is_unchanged(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])

        result = self.model(input_ids, targets)

        self.assertEqual(len(result), 2)
        logits, loss = result
        self.assertEqual(logits.shape, (1, 4, 32))
        self.assertIsNotNone(loss)

        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)

    def test_invalid_prefill_and_decode_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            TransformerLanguageModel(vocab_size=8, max_seq_len=0)

        with self.assertRaisesRegex(ValueError, "num_layers"):
            TransformerLanguageModel(vocab_size=8, num_layers=0)

        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.model.prefill(torch.empty((1, 0), dtype=torch.long))

        full_context = torch.arange(8).unsqueeze(0)
        _, kv_cache = self.model.prefill(full_context)

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            self.model.decode(torch.tensor([[8]]), kv_cache)

        with self.assertRaisesRegex(ValueError, r"\[batch, 1\]"):
            self.model.decode(torch.tensor([[1, 2]]), kv_cache)


class GenerateTest(unittest.TestCase):
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
        model = RecordingModel(max_seq_len=4)

        result = generate(
            "a b",
            max_new_tokens=3,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        self.assertEqual(result, "a b <gen> <gen> <gen>")
        self.assertEqual([kind for kind, _ in model.calls], [
            "prefill",
            "decode",
            "decode",
        ])
        torch.testing.assert_close(model.calls[0][1], torch.tensor([[1, 2]]))
        torch.testing.assert_close(model.calls[1][1], torch.tensor([[0]]))
        torch.testing.assert_close(model.calls[2][1], torch.tensor([[0]]))

    def test_generate_reprefills_latest_window_when_cache_is_full(self):
        model = RecordingModel(max_seq_len=4)

        generate(
            "a b c",
            max_new_tokens=3,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        self.assertEqual([kind for kind, _ in model.calls], [
            "prefill",
            "decode",
            "prefill",
        ])
        torch.testing.assert_close(
            model.calls[2][1],
            torch.tensor([[2, 3, 0, 0]]),
        )

    def test_generate_prefills_only_the_latest_prompt_window(self):
        model = RecordingModel(max_seq_len=4)

        generate(
            "a b c d e",
            max_new_tokens=1,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], "prefill")
        torch.testing.assert_close(
            model.calls[0][1],
            torch.tensor([[2, 3, 4, 5]]),
        )

    def test_zero_new_tokens_does_not_run_the_model(self):
        model = RecordingModel(max_seq_len=4)

        result = generate(
            "a b",
            max_new_tokens=0,
            model=model,
            stoi=self.stoi,
            itos=self.itos,
        )

        self.assertEqual(result, "a b")
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
