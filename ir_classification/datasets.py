import io
from typing import Callable, List

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import Vocab

_default_tokenizer = get_tokenizer("basic_english")
DEFAULT_LABEL_TRANSFORM = lambda x: x
DEFAULT_TEXT_TRANSFORM = lambda x: _default_tokenizer(x)


def create_torch_dataloader(
    dataset: data.Dataset,
    vocab: Vocab,
    label_transform: Callable = DEFAULT_LABEL_TRANSFORM,
    text_transform: Callable = DEFAULT_TEXT_TRANSFORM,
    weighted=True,
    **kwargs
):
    """Creates a Pytorch style dataloader using a dataset and a precompiled vocab.

    The dataset returns "model-ready" data.

    Args:
        dataset: The raw text dataset to be used during inference
        vocab: the premade vocabulary used to index words/phrases
        label_transform: any operation used on the datasets label output
        text_transform: operation used on the raw text sentence outputs from the data
        weighted: whether to weight the samples based on class distribution
        **kwargs: any additional kwargs used by Pytorch DataLoaders.

    Returns:
        A PyTorch DataLoader to be used during training, eval, or test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _collate_batch(batch):
        label_list, docid_list, text_list, offsets = [], [], [], [0]
        for (_label, _docid, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(
                vocab(text_transform(_text)), dtype=torch.int64
            )
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
            docid_list.append(_docid)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device), docid_list

    if weighted:
        weights = dataset.weights
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    else:
        sampler = None

    return data.DataLoader(
        dataset,
        collate_fn=_collate_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        **kwargs
    )


class TSVRawTextIterableDataset(data.IterableDataset):
    """Dataset that loads TSV data incrementally as an iterable and returns raw text.

    This dataset must be traversed in order as it only reads from the TSV file as it is called.
    Useful if the size of data is too large to load into memory at once.
    """

    def __init__(self, filepath: str, data_columns: List[int]):
        """Loads an iterator from a file.

        Args:
            filepath: location of the .tsv file
            data_columns: the columns in the .tsv that are used as feature data
        """
        self._number_of_items = _get_tsv_file_length(filepath)
        self._iterator = _create_data_from_tsv(
            filepath, data_column_indices=data_columns
        )
        self._current_position = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterator)
        self._current_position += 1
        return item

    def __len__(self):
        return self._number_of_items


class TSVRawTextMapDataset(data.Dataset):
    """Dataset that loads all TSV data into memory and returns raw text.

    This dataset provides a map interface, allowing access to any entry.
    Useful for modifying the sampling or order during training.
    """

    def __init__(self, filepath: str, data_columns: List[int]):
        """Loads .tsv structed data into memory.

        Args:
            filepath: location of the .tsv file
            data_columns: the columns in the .tsv that are used as feature data
        """
        self._records = list(
            _create_data_from_tsv(filepath, data_column_indices=data_columns)
        )
        self._weights = None

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._calculate_sample_weights()
        return self._weights

    def _calculate_sample_weights(self):
        targets = torch.tensor(
            [label if label > 0 else 0 for label, *_ in self._records]
        )
        unique, sample_counts = torch.unique(targets, return_counts=True)
        weight = 1.0 / sample_counts
        weights = torch.tensor([weight[t] for t in targets])
        return weights

    def __getitem__(self, index):
        return self._records[index]

    def __len__(self):
        return len(self._records)


def _create_data_from_tsv(data_path, data_column_indices):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            data = [row[i] for i in data_column_indices]
            yield int(row[0]), row[1], " ".join(data)


def _get_tsv_file_length(data_path):
    with io.open(data_path, encoding="utf8") as f:
        row_count = sum(1 for row in f)

    return row_count
