import random
import torch
from torch.utils.data import RandomSampler, DataLoader


def split_dataset(dataset: list, train_frac: float, val_frac: float) -> tuple:
    random.shuffle(dataset)
    dataset_len = len(dataset)
    train_number = int(train_frac * dataset_len)
    val_number = int(val_frac * dataset_len)
    train_dataset = dataset[:train_number]
    val_dataset = dataset[train_number : train_number + val_number]
    test_dataset = dataset[train_number + val_number :]
    return train_dataset, val_dataset, test_dataset


def create_minibatch_dataloader(dataset: list, batch_size: int, seed: int):
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader
