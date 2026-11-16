import pytest
import torch

from src.optimizer import RobustSGD


def make_noisy_grads(param, mean_grad, var):
    noise = torch.sqrt(var)
    return {
        param: torch.stack([
            mean_grad + noise,
            mean_grad - noise
        ])
    }


def run_rsgd(mean_grad, var, rho, lr=0.1):
    param = torch.nn.Parameter(torch.zeros_like(mean_grad))
    noisy_grads = make_noisy_grads(param, mean_grad, var)
    optimizer = RobustSGD([param], lr=lr, rho=rho)
    optimizer.step(noisy_grads=noisy_grads)
    return param.detach().clone()


def test_rho_zero_matches_sgd():
    init = torch.tensor([1, 2, 3], dtype=torch.float64)
    mean_grad = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float64)
    var = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    lr = 0.1

    p_rsgd = torch.nn.Parameter(init.clone())
    noisy_grads = make_noisy_grads(p_rsgd, mean_grad, var)
    optimizer_rsgd = RobustSGD([p_rsgd], lr=lr, rho=0)
    optimizer_rsgd.step(noisy_grads=noisy_grads)

    p_sgd = torch.nn.Parameter(init.clone())
    p_sgd.grad = mean_grad.clone()
    optimizer_sgd = torch.optim.SGD([p_sgd], lr=lr)
    optimizer_sgd.step()

    assert torch.allclose(p_rsgd, p_sgd, atol=1e-10)


def test_step_norm_decreases_with_rho():
    mean_grad = torch.tensor([1, 2, 0.5], dtype=torch.float64)
    var = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    rhos = [0, 0.1, 0.5, 1, 2, 3]
    step_norms = []
    for rho in rhos:
        direction = run_rsgd(mean_grad, var, rho)
        step_norms.append(torch.linalg.norm(direction).item())

    for i in range(len(step_norms) - 1):
        assert step_norms[i] >= step_norms[i + 1] - 1e-10


def test_large_rho_gives_zero_step():
    mean_grad = torch.tensor([1, 2, 0.5], dtype=torch.float64)
    var = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    direction = run_rsgd(mean_grad, var, rho=10)
    assert torch.allclose(direction, torch.zeros_like(direction), atol=1e-10)


def test_custom_gradients_override_parameter_grad():
    init = torch.tensor([1, 2], dtype=torch.float64)
    custom_mean = torch.tensor([0.5, -0.2], dtype=torch.float64)
    var = torch.tensor([0.1, 0.2], dtype=torch.float64)
    param = torch.nn.Parameter(init.clone())
    param.grad = torch.tensor([100.0, 100.0], dtype=torch.float64)
    noisy_grads = make_noisy_grads(param, custom_mean, var)
    optimizer = RobustSGD([param], lr=0.1, rho=0.0)
    optimizer.step(noisy_grads=noisy_grads)
    expected = init - 0.1 * custom_mean
    
    assert torch.allclose(param, expected, atol=1e-10)


def test_optimality_condition():
    mean_grad = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    var = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    rho = 0.5
    lr = 0.1
    direction = run_rsgd(mean_grad, var, rho, lr)

    assert torch.linalg.norm(direction) > 0
    uncertainty = var + 0.0002  # Equation (4): 1e-3 * mean([0.1, 0.2, 0.3]).
    denominator = torch.sqrt(torch.sum(uncertainty * direction.square()))
    residual = (mean_grad + rho * uncertainty * direction / denominator + direction / lr)

    assert torch.linalg.norm(residual) < 1e-8


def test_default_step_matches_torch_sgd():
    initial = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    gradient = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float64)
    lr = 0.1
    p_rsgd = torch.nn.Parameter(initial.clone())
    p_rsgd.grad = gradient.clone()

    optimizer_rsgd = RobustSGD([p_rsgd], lr=lr, rho=2.0)
    optimizer_rsgd.step()
    p_sgd = torch.nn.Parameter(initial.clone())
    p_sgd.grad = gradient.clone()
    optimizer_sgd = torch.optim.SGD([p_sgd], lr=lr)

    optimizer_sgd.step()
    assert torch.allclose(p_rsgd, p_sgd, atol=1e-10)


def test_ridge_is_shared_across_all_parameters():
    first = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    noisy_grads = {
        first: torch.tensor([[-100.0], [100.0]], dtype=torch.float64),
        second: torch.ones(2, 1, dtype=torch.float64),
    }
    optimizer = RobustSGD([first, second], lr=0.1, rho=1.0)
    optimizer.step(noisy_grads=noisy_grads)

    # Raw variances [10000, 0] give a shared ridge of 5 and g^T S^-1 g = 0.2.
    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(second, torch.zeros_like(second))


def test_ridge_floor_with_constant_observations():
    mean_grad = torch.tensor([2e-6], dtype=torch.float64)
    direction = run_rsgd(mean_grad, torch.zeros_like(mean_grad), rho=1.0)

    # Zero raw variance gives S=1e-11 and g^T S^-1 g=0.4, so the step is zero.
    assert torch.equal(direction, torch.zeros_like(direction))


def test_single_gradient_sample_raises_error():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    noisy_grads = {
        parameter: torch.randn(1, 2, dtype=torch.float64)
    }

    optimizer = RobustSGD([parameter], lr=0.1, rho=1.0)

    with pytest.raises(ValueError):
        optimizer.step(noisy_grads=noisy_grads)    
