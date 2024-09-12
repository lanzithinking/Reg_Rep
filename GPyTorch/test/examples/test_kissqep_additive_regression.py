#!/usr/bin/env python3

import os
import random
import unittest

import torch
from torch import optim

import sys
sys.path.insert(0,'../..')
import gpytorch
from gpytorch.distributions import MultivariateQExponential
from gpytorch.kernels import AdditiveStructureKernel, GridInterpolationKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import QExponentialLikelihood
from gpytorch.means import ZeroMean

POWER = 1.0

n = 20
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])
train_y = train_y + torch.randn_like(train_y).div_(20.0)

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_y = torch.sin(test_x[:, 0]) + torch.cos(test_x[:, 1])


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class QEPRegressionModel(gpytorch.models.ExactQEP):
    def __init__(self, train_x, train_y, likelihood):
        super(QEPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))
        self.covar_module = AdditiveStructureKernel(
            GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1), num_dims=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.likelihood.power)


class TestKISSQEPAdditiveRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_kissqep_qep_mean_abs_error(self, toeplitz=False):
        likelihood = QExponentialLikelihood(power=torch.tensor(POWER))
        qep_model = QEPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, qep_model)

        with gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.max_cg_iterations(50):
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(toeplitz):
                # Optimize the model
                qep_model.train()
                likelihood.train()

                optimizer = optim.Adam(qep_model.parameters(), lr=0.01)
                optimizer.n_iter = 0
                for _ in range(15):
                    optimizer.zero_grad()
                    output = qep_model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.n_iter += 1
                    optimizer.step()

                    for param in qep_model.parameters():
                        self.assertTrue(param.grad is not None)
                        self.assertGreater(param.grad.norm().item(), 0)

                # Test the model
                qep_model.eval()
                likelihood.eval()

                test_preds = likelihood(qep_model(test_x)).mean
                mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
                self.assertLess(mean_abs_error.squeeze().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
