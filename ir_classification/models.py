from torch import nn
import torch


class PretrainedEmbeddingMLPModel(nn.Module):
    """Hidden Layer model based on fasttext using pre-trained embedding weights.

    Handles variable length sentences by using offsets.
    """

    def __init__(self, num_class: int, hidden_layers: int, embedding_vectors: torch.Tensor):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embedding_vectors)
        self.hidden_layer = nn.Linear(self.embedding.embedding_dim, hidden_layers)
        self.fc = nn.Linear(hidden_layers, num_class)
        self._init_weights()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets).float()
        hidden_x = self.hidden_layer(embedded)
        return self.fc(hidden_x)

    def _init_weights(self):
        initrange = 0.5
        self.hidden_layer.weight.data.uniform_(-initrange, initrange)
        self.hidden_layer.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

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
        embedded = self.embedding(text, offsets).float()
        return self.fc(embedded)

    def _init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
