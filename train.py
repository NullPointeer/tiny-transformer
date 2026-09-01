import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. Read training data
# -----------------------------

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))


# -----------------------------
# 2. Create vocabulary
# -----------------------------

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)


# Convert text into numbers
data = torch.tensor(encode(text), dtype=torch.long)

print("Vocabulary size:", vocab_size)


# -----------------------------
# 3. Train / validation split
# -----------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# -----------------------------
# 4. Create batches
# -----------------------------

batch_size = 32
block_size = 64


def get_batch(data):
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


# -----------------------------
# 5. Simple Bigram Model
# -----------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, x, targets=None):

        # x shape:
        # (batch, sequence)

        logits = self.token_embedding(x)

        # logits shape:
        # (batch, sequence, vocab_size)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss


# -----------------------------
# 6. Create model
# -----------------------------

model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# -----------------------------
# 7. Training
# -----------------------------

steps = 5000

for step in range(steps):

    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 500 == 0:
        print(
            f"step {step}: loss {loss.item():.4f}"
        )


# -----------------------------
# 8. Generate text
# -----------------------------

context = torch.zeros(
    (1, 1),
    dtype=torch.long
)

for _ in range(500):

    logits, _ = model(context)

    # Take predictions for last character
    logits = logits[:, -1, :]

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    next_char = torch.multinomial(
        probabilities,
        num_samples=1
    )

    context = torch.cat(
        (context, next_char),
        dim=1
    )


print("\n----------------")
print(decode(context[0].tolist()))