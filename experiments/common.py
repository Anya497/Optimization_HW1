import torch
import random


def setup_seed(seed: int):
    torch.random.manual_seed(seed)
    random.seed(seed)
