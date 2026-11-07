import random

import torch

from tokenizer import SimpleTokenizer
from model import TransformerLanguageModel


batch_size = 16
seq_len = 8
d_model = 128
num_heads = 4
num_layers = 3
learning_rate = 3e-4
train_steps = 5000


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = SimpleTokenizer(text)

print("vocab size:", tokenizer.vocab_size)


data = tokenizer.encode(text)
data = torch.tensor(data, dtype=torch.long)

print("number of tokens:", len(data))


def get_batch():
    xs = []
    ys = []

    for _ in range(batch_size):
        start = random.randint(
            0,
            len(data) - seq_len - 2
        )

        x = data[start:start + seq_len]
        y = data[start + 1:start + seq_len + 1]

        xs.append(x)
        ys.append(y)

    x = torch.stack(xs)
    y = torch.stack(ys)

    return x.to(device), y.to(device)


model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=seq_len,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)


num_parameters = sum(
    p.numel() for p in model.parameters()
)

print("parameters:", num_parameters)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


model.train()

for step in range(train_steps):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"step={step:5d} "
            f"loss={loss.item():.4f}"
        )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    },
    "model.pt",
)

print("model saved")
