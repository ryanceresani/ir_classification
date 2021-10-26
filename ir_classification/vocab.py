import io
from typing import List
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab as vocab_builder
from torchtext.vocab import GloVe


def create_glove_with_unk_vector() -> GloVe:
    glove = GloVe()
    # Load the average vector for this glove embedding set to use for defaults.
    average_glove_vector = np.load("../datasets/glove_default_vector.npy")
    unk_init_vec = torch.from_numpy(average_glove_vector)
    # Extend the glove vectors with one for "unk"
    glove.vectors = torch.cat((glove.vectors, unk_init_vec.unsqueeze(0)))

    return glove

def create_vocab_from_glove(glove: GloVe):
    # Since glove is already ordered and not a counter, we overload the
    # Constructor to align the indices.
    unk_token = "<unk>"
    vocab = vocab_builder(glove.stoi, min_freq=0)
    vocab.append_token(unk_token)
    vocab.set_default_index(vocab[unk_token])

    return vocab

def create_vocab_from_tsv(
    filepath: str,
    column_indices_to_use: List[int],
    minimum_word_freq: int = 1,
    ngrams: int = 1,
):
    """Creates a PyTorch vocab object from a TSV file.

    The resulting vocab object converts words to indices for assisting in embedding and DL operations.

    Args:
        filepath: The location of the TSV file
        minimum_word_freq: How many times a word must appear to be included
        ngrams: The size of ngrams to use for the vocab
        column_indices_to_use: Which columns from the TSV are part of the actual feature set

    Returns:
        A torchtext vocab object.
    """
    unk_token = "<unk>"
    vocab = build_vocab_from_iterator(
        _tsv_iterator(filepath, ngrams=ngrams, column_indices=column_indices_to_use),
        min_freq=minimum_word_freq,
        specials=[unk_token],
    )
    vocab.set_default_index(vocab[unk_token])
    return vocab


def _tsv_iterator(data_path, ngrams, column_indices):
    # Spacy has novel tokenizer
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            row_iter = [row[i] for i in column_indices]
            tokens = " ".join(row_iter)
            yield ngrams_iterator(tokenizer(tokens), ngrams)
