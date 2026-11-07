import torch

from model import TransformerLanguageModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_default_runtime = None


def load_runtime(checkpoint_path="model.pt"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    itos = checkpoint["itos"]
    stoi = checkpoint["stoi"]

    model = TransformerLanguageModel(
        vocab_size=len(itos),
        max_seq_len=8,
        d_model=128,
        num_heads=4,
        num_layers=3,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, stoi, itos


def _get_default_runtime():
    global _default_runtime

    if _default_runtime is None:
        _default_runtime = load_runtime()

    return _default_runtime


def encode(text, stoi=None):
    if stoi is None:
        _, stoi, _ = _get_default_runtime()

    return [
        stoi.get(word, stoi["<unk>"])
        for word in text.lower().split()
    ]


def decode(ids, itos=None):
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
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    runtime_values = (model, stoi, itos)

    if all(value is None for value in runtime_values):
        model, stoi, itos = _get_default_runtime()
    elif any(value is None for value in runtime_values):
        raise ValueError("model, stoi, and itos must be provided together")

    ids = encode(prompt, stoi)
    model_device = next(model.parameters()).device

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=model_device,
    )

    if max_new_tokens == 0:
        return decode(x[0].tolist(), itos)

    if not ids:
        raise ValueError("prompt must contain at least one token")

    # Prefill processes the prompt once and builds the KV cache.
    x_context = x[:, -model.max_seq_len:]
    logits, kv_cache = model.prefill(x_context)
    cached_length = x_context.size(1)

    for step in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        x = torch.cat(
            [x, next_token],
            dim=1,
        )

        if step == max_new_tokens - 1:
            break

        if cached_length < model.max_seq_len:
            # Decode only the newly generated token while the cache has room.
            logits, kv_cache = model.decode(next_token, kv_cache)
            cached_length += 1
        else:
            # Learned absolute positions must be reset for a sliding window,
            # so rebuild the cache instead of dropping only its oldest entry.
            x_context = x[:, -model.max_seq_len:]
            logits, kv_cache = model.prefill(x_context)
            cached_length = x_context.size(1)

    return decode(x[0].tolist(), itos)


if __name__ == "__main__":
    print(generate("the cat", max_new_tokens=15))
    print(generate("the model", max_new_tokens=15))
    print(generate("attention", max_new_tokens=15))
