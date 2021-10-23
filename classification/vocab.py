import io
from typing import List

from torchtext.legacy.vocab import build_vocab_from_iterator
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader


def create_vocab_from_tsv(filepath: str, minimum_word_freq: int, ngrams: int, column_indices_to_use: List[int]):
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
    unk_token = '<unk>'
    vocab = build_vocab_from_iterator(_tsv_iterator(filepath, ngrams=ngrams, column_indices=column_indices_to_use), min_freq=minimum_word_freq, specials=[unk_token])
    vocab.set_default_index(vocab[unk_token])
    return vocab

def _tsv_iterator(data_path, ngrams, column_indices):
    # Spacy has novel tokenizer
    tokenizer = get_tokenizer("spacy")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            row_iter = [row[i] for i in column_indices]
            tokens = ' '.join(row_iter)
            yield ngrams_iterator(tokenizer(tokens), ngrams)