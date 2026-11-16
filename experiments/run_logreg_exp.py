from src.data.log_regression import get_cls_data
from torch.nn import Linear
from experiments.common import setup_seed


def main():
    seed = 1024
    setup_seed(seed)
    n_points = 1000
    in_features = 2
    out_features = 1
    dataset = get_cls_data(n_points, features_num=in_features)
    model = Linear(in_features, out_features, bias=True)
