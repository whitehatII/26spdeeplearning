import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        embed = self.embedding(x)      # (B, T, D)
        output, _ = self.lstm(embed)   # (B, T, H)

        output = output.contiguous().view(-1, output.shape[2])  # (B*T, H)

        output = self.fc(output)
        output = self.log_softmax(output)

        return output