from src.data.common import make_loaders
from src.data.log_regression import sample_points
from torch.nn import Linear
from experiments.common import setup_seed


def main():
    seed = 1024
    setup_seed(seed)
    n_points = 1000
    in_features = 2
    out_features = 1
    loaders = make_loaders(sample_points, train_size=n_points, data_seed=seed, seed=seed)
    model = Linear(in_features, out_features, bias=True)
    return model, loaders
