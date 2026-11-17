import torch
from torch.utils.data import RandomSampler, DataLoader, TensorDataset


def create_minibatch_dataloader(dataset, batch_size: int, seed: int, updates: int = 1):
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=updates * batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader


def make_dataset(size: int, sampler, seed: int) -> TensorDataset:
    """Seed sampler(size, generator), then shuffle and package its points and labels."""
    generator = torch.Generator().manual_seed(seed)
    points, labels = sampler(size, generator)
    order = torch.randperm(size, generator=generator)
    return TensorDataset(points[order], labels[order])


def make_loaders(
    sampler,
    train_size: int = 1000,
    validation_size: int = 400,
    test_size: int = 400,
    batch_size: int = 32,
    updates: int = 1000,
    data_seed: int = 42,
    seed: int = 0,
) -> dict[str, DataLoader]:
    train = make_dataset(train_size, sampler, data_seed)
    validation = make_dataset(validation_size, sampler, data_seed + 1)
    test = make_dataset(test_size, sampler, data_seed + 2)
    return {
        "train": create_minibatch_dataloader(train, batch_size, seed, updates),
        "train_eval": DataLoader(train, batch_size=256),
        "validation": DataLoader(validation, batch_size=256),
        "test": DataLoader(test, batch_size=256),
    }
