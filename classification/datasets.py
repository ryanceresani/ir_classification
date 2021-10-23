import io
from typing import Callable, List

import torch
from torch.utils import data
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import Vocab

_default_tokenizer = get_tokenizer("basic_english")
DEFAULT_LABEL_PIPELINE = lambda x: x
DEFAULT_TEXT_PIPELINE = lambda x: _default_tokenizer(x)


def create_torch_dataloader(
    dataset: data.Dataset,
    vocab: Vocab,
    label_pipeline: Callable = DEFAULT_LABEL_PIPELINE,
    text_pipeline: Callable = DEFAULT_TEXT_PIPELINE,
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(
                vocab(text_pipeline(_text)), dtype=torch.int64
            )
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    return data.DataLoader(dataset, collate_fn=_collate_batch, **kwargs)


class TSVRawTextIterableDataset(data.IterableDataset):
    def __init__(self, filepath: str, data_columns: List[int]):
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
    def __init__(self, filepath: str, data_columns: List[int]):
        self._records = [
            record
            for record in _create_data_from_tsv(
                filepath, data_column_indices=data_columns
            )
        ]

    def __getitem__(self, index):
        return self._records[index]

    def __len__(self):
        return len(self._records)


def _create_data_from_tsv(data_path, data_column_indices):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            data = [row[i] for i in data_column_indices]
            yield int(row[0]), " ".join(data)


def _get_tsv_file_length(data_path):
    with io.open(data_path, encoding="utf8") as f:
        row_count = sum(1 for row in f)

    return row_count
