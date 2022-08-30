
# %%
import math
import numpy as np
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS
from botorch.test_functions import Ackley
from botorch.optim import optimize_acqf
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import t_batch_mode_transform
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior, MCMC_DIM
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Distance

# %%
"""Run the No-U-Turn sampler"""
def sampler(model, warmup_steps, num_samples, thinning):
  model.train()
  nuts = NUTS(
    model.pyro_model.sample,
    jit_compile=True,
    full_mass=True,
    ignore_jit_warnings=True
  )
  mcmc = MCMC(
    nuts,
    warmup_steps=warmup_steps,
    num_samples=num_samples
  )
  mcmc.run()
  samples = mcmc.get_samples()
  # Thinning
  for k, v in samples.items():
    samples[k] = v[::thinning]
  model.load_samples(samples)
  model.eval()

"""Matérn kernel (without output-scale)"""
def matern52(X, lengthscale):
  nu = 5/2
  dist = Distance()._dist(
    X / lengthscale,
    X / lengthscale,
    postprocess=False,
    x1_eq_x2=True
  )
  f1 = torch.exp(-math.sqrt(nu*2)*dist)
  f2 = (math.sqrt(5)*dist).add(1).add(5.0/3.0*(dist**2))
  return f1*f2

"""Sample average acquisition function"""
class fbEI(ExpectedImprovement):
  @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
  def forward(self, X):
    ei = super().forward(X)
    return ei.mean(-1)

"""Pyro model"""
class FboPyro():
  def __init__(self, priors):
    self.priors = priors

  def set_inputs(self, train_X, train_Y, train_Yvar):
    self.train_X = train_X
    self.train_Y = train_Y
    self.train_Yvar = train_Yvar

  def sample(self):
    mean = pyro.param("mean", torch.tensor([0.0])) # zero-mean
    noise = self.train_Yvar # noise variance
    outputscale = pyro.sample(
      "outputscale",
      pyro.distributions.LogNormal(
        torch.tensor(self.priors["outputscale"]["p1"]),
        torch.tensor(self.priors["outputscale"]["p2"]),
      )
    )
    lengthscale = pyro.sample(
      "lengthscale",
      pyro.distributions.LogNormal(
        torch.tensor(self.priors["lengthscale"]["p1"]),
        torch.tensor(self.priors["lengthscale"]["p2"]),
      )
    )
    k = matern52(X=self.train_X, lengthscale=lengthscale)
    cov = outputscale * k + noise * torch.eye(self.train_X.shape[0])
    pyro.sample(
      "Y",
      pyro.distributions.MultivariateNormal(
        loc=mean.view(-1).expand(self.train_X.shape[0]),
        covariance_matrix=cov,
      ),
      obs=self.train_Y.squeeze(-1)
    )

  def load_samples(self, samples):
    def reshape_and_detach(target, new_value):
      return new_value.detach().clone().view(target.shape).to(target)

    num_samples = samples["lengthscale"].shape[0]
    batch_shape = torch.Size([num_samples])
    covar_module = ScaleKernel(
      base_kernel=MaternKernel(
        ard_num_dims=len(self.priors["lengthscale"]["p1"]),
        batch_shape=batch_shape
      ),
      batch_shape=batch_shape
    )
    covar_module.base_kernel.lengthscale = reshape_and_detach(
      target=covar_module.base_kernel.lengthscale,
      new_value=samples["lengthscale"]
    )
    covar_module.outputscale = reshape_and_detach(
      target=covar_module.outputscale,
      new_value=samples["outputscale"]
    )
    return covar_module

"""GP"""
class FboGP(SingleTaskGP):
  def __init__(self, train_X, train_Y, train_Yvar, priors):
    super().__init__(train_X, train_Y)
    self.likelihood.noise = train_Yvar[0]
    pyro_model = FboPyro(priors=priors)
    pyro_model.set_inputs(
      train_X=train_X,
      train_Y=train_Y,
      train_Yvar=train_Yvar
    )
    self.pyro_model = pyro_model

  def forward(self, X):
    return super().forward(X.unsqueeze(MCMC_DIM)) # MCMC_DIM=-3

  def posterior(self,
    X,
    output_indices=None,
    observation_noise=False,
    posterior_transform=None
  ):
    posterior = super().posterior(
      X=X,
      output_indices=output_indices,
      observation_noise=observation_noise,
      posterior_transform=posterior_transform
    )
    posterior = FullyBayesianPosterior(mvn=posterior.mvn)
    return posterior

  def load_samples(self, samples):
    self.covar_module = self.pyro_model.load_samples(samples=samples)


# %%
"""Run"""
# if __name__... is just to indicate the beginning of process
if __name__ == "__main__":
  seed = 1 # random seed

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
  WARMUP_STEPS = 512 # MCMC
  NUM_SAMPLES = 256 # MCMC
  THINNING = 16 # MCMC
  NUM_RESTARTS = 128 # multistart acquisition optim
  E_l = 0.5 # mean of length-scale
  E_o = 10.0 # mean of output-scale
  N_ITER = 30 # maximum evaluation budget
  # Priors (n.b. relative std = sig/mu = 1)
  sig_l = np.full(DIM, np.sqrt(np.log((E_l/E_l)**2+1)))
  mu_l = np.log(E_l) - sig_l**2/2
  sig_o = np.sqrt(np.log((E_o/E_o)**2+1))
  mu_o = np.log(E_o) - sig_o**2/2
  # p1: location, p2: scale
  priors = {"lengthscale": {"p1": mu_l, "p2": sig_l},
            "outputscale": {"p1": mu_o, "p2": sig_o}}

  """Initial observation"""
  # Keep both raw Y and standardised Y (train_Y)
  # Keep only normalised X. Un-normalisation is handled by f(x).
  # At n=0, Y.std()=0. So, set train_Y=0.
  X0 = torch.tensor(X0s[seed]).unsqueeze(0) # seed selects a row of X0
  X = torch.clone(X0)
  Y = f(X).unsqueeze(-1)
  train_Y = Y - Y.mean()

  """BO steps"""
  pyro.set_rng_seed(seed) # set a seed in torch, random, and numpy

  for iter in range(N_ITER):
    # Update GP
    gp = FboGP(
      train_X=X,
      train_Y=train_Y,
      train_Yvar=torch.full_like(train_Y, 1e-6),
      priors=priors
    )
    # Sample hyperparameters
    sampler(
      gp,
      warmup_steps=WARMUP_STEPS,
      num_samples=NUM_SAMPLES,
      thinning=THINNING,
    )
    # Maxise acquisition
    acqf = fbEI(model=gp, best_f=train_Y.max())
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
    train_Y = -1 * (Y-Y.mean())/Y.std() # Standardisation

  print(Y.detach().numpy().squeeze())

  # %%
  import pickle
  with open("Ackley.pickle", "rb") as f:
    re = pickle.load(f)
  np.array_equal(Y.detach().numpy().squeeze(), re["data"][seed][0.5].squeeze())
