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

# Local file.
import turbolib


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
                        X_cand, y_cand, _ = turbolib._create_candidates(
                            self, X, fX, length=_length, n_training_steps=100, hypers={},
                            is_rbf= i == 0, nu=nu
                        )
                        _X_next = np.vstack((_X_next, self.turbo._select_candidates(X_cand, y_cand)[:2, :]))

                    X_next[-n_adapt:, :] = _X_next[:n_adapt, :]
                    X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)
                else:
                    X_cand = np.zeros((len(nus), self.turbo.n_cand, self.turbo.dim))
                    y_cand = np.inf * np.ones((len(nus), self.turbo.n_cand, self.turbo.batch_size))
                    for i, nu in enumerate(nus):
                        X_cand[i, :, :], y_cand[i, :, :], _ = turbolib._create_candidates(
                            self, X, fX, length=_length, n_training_steps=100, hypers={},
                            is_rbf= i == 0, nu=nu
                        )
                    _X_next = turbolib._select_candidates(self, X_cand, y_cand)

                    X_next[-n_adapt:, :] = _X_next[:n_adapt, :]
                    X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)

        # Unwarp the suggestions
        suggestions = self.space_x.unwarp(X_next)
        self._suggest_counter += 1
        return suggestions


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


if __name__ == "__main__":
    experiment_main(TurboOptimizer)
