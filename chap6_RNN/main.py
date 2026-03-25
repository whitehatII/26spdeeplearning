import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim

from rnn import RNNModel

start_token = 'G'
end_token = 'E'
pad_token = '<PAD>'

BATCH_SIZE = 64
EPOCHS = 500
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_poems(file_name):
    poems = []

    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            content = line.replace(' ', '')

            if any(c in content for c in ['_', '(', '（', '《', '[', ']']):
                continue

            if start_token in content or end_token in content:
                continue

            if len(content) < 5 or len(content) > 80:
                continue

            poems.append(start_token + content + end_token)

    poems = sorted(poems, key=lambda x: len(x))

    all_words = [w for poem in poems for w in poem]
    counter = collections.Counter(all_words)

    words, _ = zip(*counter.most_common())
    words = list(words)

    words.append(pad_token)

    word2id = {w: i for i, w in enumerate(words)}
    id2word = dict(enumerate(words))

    poems_vec = [[word2id[w] for w in poem] for poem in poems]

    print("poems count:", len(poems))

    return poems_vec, word2id, id2word


def generate_batch(poems_vec, batch_size, pad_id):
    np.random.shuffle(poems_vec)
    batches = []

    for i in range(0, len(poems_vec) - batch_size, batch_size):
        batch = poems_vec[i:i + batch_size]
        max_len = max(len(p) for p in batch)

        x_batch, y_batch = [], []

        for poem in batch:
            x = poem[:-1]
            y = poem[1:]

            x = x + [pad_id] * (max_len - len(x))
            y = y + [pad_id] * (max_len - len(y))

            x_batch.append(x)
            y_batch.append(y)

        batches.append((
            torch.LongTensor(x_batch),
            torch.LongTensor(y_batch)
        ))

    return batches


def train():
    poems_vec, word2id, id2word = process_poems('./tangshi.txt')

    pad_id = word2id[pad_token]

    model = RNNModel(
        vocab_size=len(word2id),
        embedding_dim=128,
        hidden_dim=256
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.NLLLoss(ignore_index=pad_id)

    for epoch in range(EPOCHS):
        batches = generate_batch(poems_vec, BATCH_SIZE, pad_id)

        total_loss = 0

        for batch_x, batch_y in batches:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            output = model(batch_x)  # (B*T, vocab)
            loss = loss_fn(output, batch_y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(batches):.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "poem_model.pth")

    return model, word2id, id2word


def generate(model, word2id, id2word, start_word, max_len=50, temperature=0.8):
    model.eval()

    # 初始输入
    input_ids = [word2id[start_token]]

    # 起始字作为上下文（不是直接拼接）
    for w in start_word:
        input_ids.append(word2id.get(w, 0))

    result = ""

    for _ in range(max_len):
        x = torch.LongTensor([input_ids]).to(DEVICE)

        with torch.no_grad():
            output = model(x)

        logits = output[-1] / temperature
        prob = torch.softmax(logits, dim=0)

        next_id = torch.multinomial(prob, 1).item()
        next_word = id2word[next_id]

        if next_word == end_token:
            break

        result += next_word
        input_ids.append(next_id)

    return start_word + result


if __name__ == "__main__":
    model, word2id, id2word = train()

    for w in ["日", "红", "山", "夜", "湖", "海", "月"]:
        poem = generate(model, word2id, id2word, w)
        print(poem)
        print()