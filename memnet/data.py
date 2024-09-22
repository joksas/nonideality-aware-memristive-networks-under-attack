import logging
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import datasets, transforms

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100


class DatasetName(Enum):
    MNIST = auto()
    FASHION_MNIST = auto()

    def __str__(self):
        return DatasetName(self.value).name

    def pretty(self):
        match self:
            case DatasetName.MNIST:
                return "MNIST"
            case DatasetName.FASHION_MNIST:
                return "Fashion MNIST"


class Subset(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class TransformedDataset(TorchDataset):
    def __init__(self, examples: list[tuple[Any, Any]]):
        self.examples_id = store_tmp_data(examples)
        self.examples: Optional[torch.Tensor] = None
        self.retrieval_count = 0
        self.example_count = len(examples)

    def __getitem__(self, index):
        if self.examples is None:
            self.examples = load_tmp_data(self.examples_id)
            logging.debug("Loaded examples into memory")

        example = self.examples[index]

        self.retrieval_count += 1
        if self.retrieval_count == len(self.examples):
            del self.examples
            self.retrieval_count = 0
            logging.debug("Cleared examples from memory")

        return example

    def __len__(self):
        return self.example_count


class Dataset:
    def __init__(self, name: DatasetName, train_val_split: float = 0.8):
        self.name = name
        self.train_val_split = train_val_split

        train_dataset = self._dataset(self.name, Subset.TRAIN)
        train_loader = self._loader(train_dataset, Subset.TRAIN)
        train_loader, val_loader = self._split_train_data(train_loader)
        self._train_loader_id = store_tmp_data(train_loader)
        self._val_loader_id = store_tmp_data(val_loader)

        test_dataset = self._dataset(self.name, Subset.TEST)
        test_loader = self._loader(test_dataset, Subset.TEST)
        self._test_loader_id = store_tmp_data(test_loader)

    def loader(self, subset: Subset) -> DataLoader:
        if subset == Subset.TRAIN:
            return load_tmp_data(self._train_loader_id)
        elif subset == Subset.VAL:
            return load_tmp_data(self._val_loader_id)
        elif subset == Subset.TEST:
            return load_tmp_data(self._test_loader_id)

    @property
    def train_loader(self) -> DataLoader:
        return self.loader(Subset.TRAIN)

    @property
    def val_loader(self) -> DataLoader:
        return self.loader(Subset.VAL)

    @property
    def test_loader(self) -> DataLoader:
        return self.loader(Subset.TEST)

    def set_loader(self, subset: Subset, new_loader: DataLoader) -> None:
        if subset == Subset.TRAIN:
            self._train_loader_id = store_tmp_data(new_loader)
        elif subset == Subset.VAL:
            self._val_loader_id = store_tmp_data(new_loader)
        elif subset == Subset.TEST:
            self._test_loader_id = store_tmp_data(new_loader)

    def units(self) -> tuple[int, int]:
        if self.name == DatasetName.MNIST:
            return 28 * 28, 10
        elif self.name == DatasetName.FASHION_MNIST:
            return 28 * 28, 10

        raise NotImplementedError(f"Unknown dataset: {self}")

    def _dataset(
        self, dataset_name: DatasetName, subset: Literal[Subset.TRAIN, Subset.TEST]
    ) -> TorchDataset:
        if dataset_name == DatasetName.MNIST:
            return datasets.MNIST(
                root="~/.pytorch-datasets",
                train=subset == Subset.TRAIN,
                download=True,
                transform=transforms.ToTensor(),
            )
        elif dataset_name == DatasetName.FASHION_MNIST:
            return datasets.FashionMNIST(
                root="~/.pytorch-datasets",
                train=subset == Subset.TRAIN,
                download=True,
                transform=transforms.ToTensor(),
            )

        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    def _loader(self, dataset_: TorchDataset, subset: Subset) -> DataLoader:
        if subset == Subset.TRAIN:
            return DataLoader(
                dataset_,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
            )
        elif subset == Subset.TEST:
            return DataLoader(
                dataset_,
                batch_size=TEST_BATCH_SIZE,
            )

        raise NotImplementedError(f"Unknown subset: {subset}")

    def _split_train_data(
        self, data_loader: DataLoader, split: float = 0.8
    ) -> tuple[DataLoader, DataLoader]:
        """Split the training data into a smaller training set and a validation set."""
        dataset_size = len(data_loader.dataset)
        train_set_size = int(split * dataset_size)
        val_set_size = dataset_size - train_set_size

        train_set, val_set = data.random_split(
            data_loader.dataset, [train_set_size, val_set_size]
        )

        train_loader = DataLoader(
            train_set,
            batch_size=data_loader.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=data_loader.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader


def transform_loader(
    examples: list[tuple[Any, Any]],
    subset: Subset,
) -> DataLoader:
    dataset = TransformedDataset(examples)
    if subset == Subset.TRAIN:
        return DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )
    if subset == Subset.VAL:
        return DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )
    elif subset == Subset.TEST:
        return DataLoader(dataset, batch_size=TEST_BATCH_SIZE)


def store_tmp_data(data: Any) -> uuid.UUID:
    id = uuid.uuid4()
    path = Path(f".tmp/{id}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return id


def load_tmp_data(id: uuid.UUID) -> Any:
    path = Path(f".tmp/{id}.pt")
    return torch.load(path)
