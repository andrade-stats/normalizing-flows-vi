
import torch
import commons

import numpy as np
import synthetic_data
from normalizing_flows_core import FlowsMixture, train
from target_distributions import BayesianLinearRegressionSimple, HorseshoeRegression, ConjugateBayesianLinearRegression, Funnel, BayesianLasso, HorseshoePriorLogisticRegression, MultivariateNormalMixture, MultivariateStudentT
import estimators

# *************** Set up computational resources used by PyTorch *************

commons.DATA_TYPE = "double" # recommend using double instead of float
commons.setGPU()  # sets GPU if available otherwise uses CPU
torch.manual_seed(432432)


# *************** Specify Bayesian Model / Target Distribution *************

np.random.seed(432432) # used only for the synthetic data generation

DATA_SAMPLES = 10
DATA_DIM = 10
X, y, true_beta , _ = synthetic_data.lasso_linear(n = DATA_SAMPLES, d = DATA_DIM)
X, y = commons.get_pytorch_tensors(X, y)

# p(theta, D)
target = BayesianLinearRegressionSimple(X, y, prior_variance = 1.0, likelihood_variance = 1.0)

# target = HorseshoeRegression(X, y) # n = 1000, data_dim = 1000 : around 4 hours with Ada GPU
# target = ConjugateBayesianLinearRegression(X, y)
# target = BayesianLasso(X, y)
# target = Funnel(10)
# target = MultivariateNormalMixture(10)
# target = MultivariateNormalMixture(10)
# target = MultivariateStudentT(10)

# rng = np.random.default_rng(293309)
# X, y, true_beta , _ , _ = synthetic_data.get_sample_data_logistic_regression(rng, data_size = DATA_SAMPLES, data_dim = DATA_DIM)
# X, y = commons.get_pytorch_tensors(X, y)
# target = HorseshoePriorLogisticRegression(X, y)


# *************** Specify Variational Approximation Family and Optimization Parameters *************

# VARIATIONAL_APPROXIMATION_TYPE = "RealNVP"
VARIATIONAL_APPROXIMATION_TYPE = "GAUSSIAN_MEAN_FIELD"
# VARIATIONAL_APPROXIMATION_TYPE = "GAUSSIAN_MIXTURE"

if VARIATIONAL_APPROXIMATION_TYPE == "RealNVP":
    nr_mixture_components = 1
    flow_type = "RealNVP"
    number_of_flows = 64
    initial_loc_spec = "zeros"
    use_student_base = True
    use_LOFT = True
    hidden_layer_size_spec = 100
    learn_mixture_weights = False   # not relevant here
elif VARIATIONAL_APPROXIMATION_TYPE == "GAUSSIAN_MEAN_FIELD":
    # vanilla mean field approximation
    nr_mixture_components = 1
    flow_type = "GaussianOnly"
    initial_loc_spec = "zeros"
    learn_mixture_weights = False  # not relevant here
    number_of_flows = 0 # not relevant here
    use_student_base = False # not relevant here
    use_LOFT = False # not relevant here
    hidden_layer_size_spec = None # not relevant here
elif VARIATIONAL_APPROXIMATION_TYPE == "GAUSSIAN_MIXTURE":
    # gaussian mixture, starting training with randomly initialized locations
    nr_mixture_components = 3
    flow_type = "GaussianOnly"
    initial_loc_spec = "random_small"
    learn_mixture_weights = True
    number_of_flows = 0 # not relevant here
    use_student_base = False # not relevant here
    use_LOFT = False # not relevant here
    hidden_layer_size_spec = None # not relevant here
else:
    assert(False)

vi_approx = FlowsMixture(target, nr_mixture_components, flow_type, number_of_flows, learn_mixture_weights, initial_loc_spec, use_student_base, use_LOFT, hidden_layer_size_spec)


# *************** Run Training of Normalizing Flow and Evaluation *************

MAX_ITERATIONS = 50000  # this is problem (dimension) dependent, recommend >= 50000
LEARNING_RATE = 10 ** (-4) # recommended default value
DIVERGENCE = "reverse_kld_without_score" # recommend using reverse KLD without score (sometimes referred to as Path Gradients)

# specify name of file where model is saved in "all_trained_models/"
commons.INFO_STR = target.__class__.__name__ + "_" + VARIATIONAL_APPROXIMATION_TYPE + "_" + str(MAX_ITERATIONS) + "maxit_" + str(DATA_SAMPLES) + "_" + str(DATA_DIM) + "synthetic_data"
print("commons.INFO_STR = ", commons.INFO_STR)

# train normalizing flow
nr_optimization_steps, best_true_loss = train(vi_approx, max_iter = MAX_ITERATIONS, learning_rate = LEARNING_RATE, divergence = DIVERGENCE)
print("nr_optimization_steps = ", nr_optimization_steps)
print("best_true_loss = ", best_true_loss)

# load normalizing flow (the one with minimal training loss)
vi_approx.load_state_dict(torch.load(commons.get_model_filename_best(), map_location = commons.DEVICE))
vi_approx.eval()

# use samples form normalizing flow for estimating marginal likelihood etc.
elbo = estimators.elbo_estimate(vi_approx, num_samples = 20000)
print("lower bound on marginal likelihood = ", elbo)

mll = estimators.importance_sampling(vi_approx, num_samples = 20000)
print("marginal likelihood estimate (with importance samples)= ", mll)

print("true marinal likelihood = ", target.true_log_marginal)

posterior_samples, _, _ = estimators.get_posterior_samples(vi_approx)
beta_samples = posterior_samples["beta"]

beta_posterior_mean = np.mean(beta_samples, axis = 0)
print("E[beta | D] = ", beta_posterior_mean)
print("true_beta = ", true_beta)