import collections
from copy import deepcopy
import itertools

import numpy as np
import scipy.stats as ss
import torch
from turbo import Turbo1
from turbo.utils import from_unit_cube, to_unit_cube

import bayesmark
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace

import math

# from gp.py
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
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
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


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def sobol_sequence(n_pts, sobol_engine):
    return sobol_engine.draw(n_pts).to(dtype=torch.float64, device=torch.device("cpu")).cpu().detach().numpy()


class TurboOptimizer(AbstractOptimizer):
    primary_import = "Turbo"

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []

        self.turbo = Turbo1(
            f=None,
            lb=self.bounds[:, 0],
            ub=self.bounds[:, 1],
            n_init=2 * self.dim + 1,
            max_evals=self.max_evals,
            batch_size=1,  # We need to update this later
            verbose=False,
        )

        # todo: add some option or config to switch grid search
        self.enable_grid_search = True
        self._grid_suggestion = False
        if self.enable_grid_search:
            self.init_grid()

        self.sobol_engine = torch.quasirandom.SobolEngine(self.turbo.dim, scramble=True, seed=None)
        self.create_mask(api_config)
        self._suggest_counter = 0

        # Configuration to search using optuna
        self._select_cand_kernelwise = False
        # self._use_rbf_kernel = True
        # self._use_matern_1_2 = True
        # self._use_matern_3_2 = True
        # self._use_matern_5_2 = True
        # self._use_length_mask = True

    def create_mask(self, api_config):
        self._length_mask = []
        for _, conf in api_config.items():
            param_type = conf["type"]
            param_space = conf.get("space", None)
            if param_type in {"bool", "cat"}:
                self._length_mask.append(self.turbo.length_max)
            elif param_type in {"real", "int"} and param_space == "logit":
                self._length_mask.append(self.turbo.length_max)
            else:
                self._length_mask.append(self.turbo.length_min)
        self._length_mask = np.array(self._length_mask)

    def init_grid(self):
        use_grid = True
        self.grid_keys = None
        self.grids = None
        self.grid_id = 0
        param_value = collections.OrderedDict()
        for param in self.space_x.param_list:
            space = self.space_x.spaces[param]
            print(space)
            if isinstance(space, bayesmark.space.Integer):
                param_value[param] = list(range(space.lower, space.upper + 1))
            elif isinstance(space, bayesmark.space.Categorical):
                param_value[param] = list(space.values)
            elif isinstance(space, bayesmark.space.Boolean):
                param_value[param] = [True, False]
            else:
                use_grid = False
                break

        if use_grid:
            n_grids = 1
            for v in param_value.values():
                n_grids *= len(v)

            if n_grids <= 8 * 16:
                self.grid_keys = list(param_value.keys())
                self.grids = list(itertools.product(*param_value.values()))

    def get_grid_suggestions(self, n_suggestions):
        self._grid_suggestion = True
        suggestions = []
        for _ in range(n_suggestions):
            if self.grid_id >= len(self.grids):
                _n_suggestions = n_suggestions- len(suggestions)
                suggestions += bayesmark.random_search.suggest_dict([], [], self.api_config, n_suggestions=_n_suggestions)
                return suggestions

            suggestion = {}
            grid = self.grids[self.grid_id]
            for i, k in enumerate(self.grid_keys):
                suggestion[k] = grid[i]
            self.grid_id += 1
            suggestions.append(suggestion)
        return suggestions

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        X_init = sobol_sequence(self.turbo.n_init, self.sobol_engine)
        self.X_init = from_unit_cube(X_init, self.lb, self.ub)
        self._suggest_counter = 0

    def suggest(self, n_suggestions=1):
        try:
            v = self._suggest(n_suggestions)
        except Exception as e:
            import sys
            import traceback
            print("Exception:", e)
            print("Stacktrace:")
            stacktrace = '\n'.join(traceback.format_tb(e.__traceback__))
            print(stacktrace)
            sys.exit(1)
        return v

    def _suggest(self, n_suggestions=1):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.turbo.batch_size = n_suggestions
            self.turbo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.turbo.n_init = max([self.turbo.n_init, self.batch_size])
            self.restart()

        if self.grid_keys is not None and self.enable_grid_search:
            return self.get_grid_suggestions(n_suggestions)

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        nus = [2.5, 2.5, 1.5, 0.5]

        if n_adapt > 0:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                X = to_unit_cube(deepcopy(self.turbo._X), self.lb, self.ub)
                fX = copula_standardize(deepcopy(self.turbo._fX).ravel())  # Use Copula

                if self._suggest_counter < 10:
                    _length = np.array([self.turbo.length] * self.turbo.dim)
                    _length = np.maximum(_length, self._length_mask)
                else:
                    _length = self.turbo.length

                if self._select_cand_kernelwise:
                    _X_next = np.zeros((0, self.turbo.dim))
                    for i, nu in enumerate(nus):
                        X_cand, y_cand, _ = self._create_candidates(
                            X, fX, length=_length, n_training_steps=100, hypers={},
                            is_rbf= i == 0, nu=nu
                        )
                        _X_next = np.vstack((_X_next, self.turbo._select_candidates(X_cand, y_cand)[:2, :]))

                    X_next[-n_adapt:, :] = _X_next[:n_adapt, :]
                    X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)
                else:
                    X_cand = np.zeros((len(nus), self.turbo.n_cand, self.turbo.dim))
                    y_cand = np.inf * np.ones((len(nus), self.turbo.n_cand, self.turbo.batch_size))
                    for i, nu in enumerate(nus):
                        X_cand[i, :, :], y_cand[i, :, :], _ = self._create_candidates(
                            X, fX, length=_length, n_training_steps=100, hypers={},
                            is_rbf= i == 0, nu=nu
                        )
                    _X_next = self._select_candidates(X_cand, y_cand)

                    X_next[-n_adapt:, :] = _X_next[:n_adapt, :]
                    X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)

        # Unwarp the suggestions
        suggestions = self.space_x.unwarp(X_next)
        self._suggest_counter += 1
        return suggestions

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (4, self.turbo.n_cand, self.turbo.dim)
        assert y_cand.shape == (4, self.turbo.n_cand, self.turbo.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.turbo.batch_size, self.turbo.dim))
        for k in range(self.turbo.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (4, self.turbo.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        if self._grid_suggestion:
            return

        assert len(X) == len(y)
        XX, yy = self.space_x.warp(X), np.array(y)[:, None]

        if len(self.turbo._fX) >= self.turbo.n_init:
            self.turbo._adjust_length(yy)

        self.turbo.n_evals += self.batch_size

        self.turbo._X = np.vstack((self.turbo._X, deepcopy(XX)))
        self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(yy)))
        self.turbo.X = np.vstack((self.turbo.X, deepcopy(XX)))
        self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(yy)))

        # Check for a restart
        if self.turbo.length < self.turbo.length_min:
            self.restart()

        # Restart if all observation is the same.
        for y1, y2 in zip(yy.tolist(), yy.tolist()[1:]):
            if y1 != y2:
               return
        self.restart()


    def _create_candidates(self, X, fX, length, n_training_steps, hypers, is_rbf=False, nu=2.5):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.turbo.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.turbo.device, self.turbo.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.turbo.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.turbo.use_ard, num_steps=n_training_steps, hypers=hypers,
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
        sobol = torch.quasirandom.SobolEngine(self.turbo.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.turbo.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.turbo.dim, 1.0)
        mask = np.random.rand(self.turbo.n_cand, self.turbo.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.turbo.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.turbo.n_cand, self.turbo.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.turbo.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.turbo.device, self.turbo.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.turbo.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.turbo.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers


if __name__ == "__main__":
    experiment_main(TurboOptimizer)
