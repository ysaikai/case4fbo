
import torch
from botorch.test_functions import Ackley
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Positive
import numpy as np

# if __name__... is just to indicate the beginning of process
if __name__ == "__main__":
  seed = 76 # random seed

  """Objective function"""
  problem = Ackley()
  DIM = problem.dim
  # Evaluate the objective using the original input scale
  def f(x):
    lb, ub = problem.bounds
    return problem(lb + (ub - lb) * x)    

  """Parameters"""
  np.random.seed(0)
  X0s = np.round(np.random.rand(101,DIM), 2) # Initial X
  NUM_RESTARTS = 128 # multistart acquisition optim
  N_ITER = 30 # maximum evaluation budget

  """Initial observation"""
  # Keep both raw Y and standardised Y (train_Y)
  # Keep only normalised X. Un-normalisation is handled by f(x).
  # At n=0, Y.std()=0. So, set train_Y=0.
  X0 = torch.tensor(X0s[seed]).unsqueeze(0) # seed selects a row of X0
  X = torch.clone(X0)
  Y = f(X).unsqueeze(-1)
  train_Y = Y - Y.mean() # = 0

  """BO steps"""
  torch.manual_seed(seed)

  for iter in range(N_ITER):
    likelihood = GaussianLikelihood(noise_constraint=Positive())
    likelihood.noise=1e-6
    likelihood.noise_covar.raw_noise.requires_grad_(False) # to fix noise
    gp = SingleTaskGP(X, train_Y, likelihood)
    gp.mean_module.constant.requires_grad_(False) # to fix mean=0
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    acqf = ExpectedImprovement(model=gp, best_f=train_Y.max())
    X_next, _ = optimize_acqf(
      acqf,
      bounds=torch.tensor([[0.]*DIM,[1.]*DIM]),
      q=1,
      num_restarts=NUM_RESTARTS,
      raw_samples=1024
    )

    Y_next = f(X_next).unsqueeze(-1)
    X = torch.cat((X, X_next))
    Y = torch.cat((Y, Y_next))
    train_Y = -1 * (Y-Y.mean())/Y.std() # Standardisation. -1 is for maximisation.
