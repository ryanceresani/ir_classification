from torch import nn


class EmbeddingBagLinearModel(nn.Module):
    """Simple Linear Model using an 'Embedding Bag'.

    Handles variable length sentences by using offsets.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self._init_weights()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
