import io
from torchtext.utils import (
    unicode_csv_reader,
)
from typing import Callable, List
from torch.utils import data

class TSVClassificationDataset(data.IterableDataset):
    def __init__(self, filepath: str, data_columns: List[int], tokenizer: Callable):
        self._number_of_items = _get_tsv_file_length(filepath)
        self._iterator = _create_data_from_tsv(filepath, data_column_indices=data_columns)
        self._tokenizer = tokenizer
        self._current_position = 0


    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterator)
        self._current_position += 1
        label, text = item
        return label+1, self._tokenizer(text)

    def __len__(self):
        return self._number_of_items

def _create_data_from_tsv(data_path, data_column_indices):
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter="\t")
        for row in reader:
            data = [row[i] for i in data_column_indices]
            yield int(row[0]), ' '.join(data)

def _get_tsv_file_length(data_path):
    with io.open(data_path, encoding="utf8") as f:
        row_count = sum(1 for _ in f)
    return row_count