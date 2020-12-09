###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################
from copy import deepcopy
import math

import numpy as np
import torch
import gpytorch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims, nu=2.5):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=nu)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class RBFGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(RBFGP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}, is_rbf=False, nu=2.5):
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    if is_rbf:
        model = RBFGP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)
    else:
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
            nu=nu,
        ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


def _select_candidates(optimizer, X_cand, y_cand):
    assert X_cand.shape == (4, optimizer.turbo.n_cand, optimizer.turbo.dim)
    assert y_cand.shape == (4, optimizer.turbo.n_cand, optimizer.turbo.batch_size)
    assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

    X_next = np.zeros((optimizer.turbo.batch_size, optimizer.turbo.dim))
    for k in range(optimizer.turbo.batch_size):
        i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (4, optimizer.turbo.n_cand))
        assert y_cand[:, :, k].min() == y_cand[i, j, k]
        X_next[k, :] = deepcopy(X_cand[i, j, :])
        assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

        # Make sure we never pick this point again
        y_cand[i, j, :] = np.inf

    return X_next


def _create_candidates(optimizer, X, fX, length, n_training_steps, hypers, is_rbf=False, nu=2.5):
    # Pick the center as the point with the smallest function values
    # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
    assert X.min() >= 0.0 and X.max() <= 1.0

    # Standardize function values.
    mu, sigma = np.median(fX), fX.std()
    sigma = 1.0 if sigma < 1e-6 else sigma
    fX = (deepcopy(fX) - mu) / sigma

    # Figure out what device we are running on
    if len(X) < optimizer.turbo.min_cuda:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = optimizer.turbo.device, optimizer.turbo.dtype

    # We use CG + Lanczos for training if we have enough data
    with gpytorch.settings.max_cholesky_size(optimizer.turbo.max_cholesky_size):
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
        gp = train_gp(
            train_x=X_torch, train_y=y_torch, use_ard=optimizer.turbo.use_ard, num_steps=n_training_steps, hypers=hypers,
            is_rbf=is_rbf, nu=nu,
        )

        # Save state dict
        hypers = gp.state_dict()

    # Create the trust region boundaries
    x_center = X[fX.argmin().item(), :][None, :]
    weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
    weights = weights / weights.mean()  # This will make the next line more stable
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
    lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

    # Draw a Sobolev sequence in [lb, ub]
    seed = np.random.randint(int(1e6))
    sobol = torch.quasirandom.SobolEngine(optimizer.turbo.dim, scramble=True, seed=seed)
    pert = sobol.draw(optimizer.turbo.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    pert = lb + (ub - lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / optimizer.turbo.dim, 1.0)
    mask = np.random.rand(optimizer.turbo.n_cand, optimizer.turbo.dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[ind, np.random.randint(0, optimizer.turbo.dim - 1, size=len(ind))] = 1

    # Create candidate points
    X_cand = x_center.copy() * np.ones((optimizer.turbo.n_cand, optimizer.turbo.dim))
    X_cand[mask] = pert[mask]

    # Figure out what device we are running on
    if len(X_cand) < optimizer.turbo.min_cuda:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = optimizer.turbo.device, optimizer.turbo.dtype

    # We may have to move the GP to a new device
    gp = gp.to(dtype=dtype, device=device)

    # We use Lanczos for sampling if we have enough data
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(optimizer.turbo.max_cholesky_size):
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([optimizer.turbo.batch_size])).t().cpu().detach().numpy()

    # Remove the torch variables
    del X_torch, y_torch, X_cand_torch, gp

    # De-standardize the sampled values
    y_cand = mu + sigma * y_cand

    return X_cand, y_cand, hypers
