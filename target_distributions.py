import torch
import commons
from normflows.distributions import Target
import torch.distributions as dist
import densities
import numpy

from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.cauchy import Cauchy
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.exponential import Exponential

import scipy.stats
import math

EPSILON = 10 ** -6 # to ensure strict positiveness

from normflows.distributions import Target
from helper import IndexManager


class BayesianLinearRegressionSimple(torch.nn.Module):

    def __init__(self, X, y, prior_variance, likelihood_variance):
        super().__init__()
    
        # Data D
        self.X =  commons.moveToDevice(X)
        self.y = commons.moveToDevice(y)
        
        self.data_dim = self.X.shape[1]
        self.n = self.X.shape[0]

        # register non-parameters (this is useful for example to ensure that their are also moved to the GPU if you call model.cuda() etc.)
        self.register_buffer('likelihood_variance', torch.tensor([likelihood_variance]))  # treated as a constant here (for simplicity)
        self.register_buffer('prior_variance', torch.tensor([prior_variance]))

        # specify dimension of each parameter
        self.idm = IndexManager(beta = self.data_dim, intercept = 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = None

        self.true_log_marginal = torch.nan # unknown

        return
    

    def log_prob(self, z):
        assert(z.shape[1] == self.d) 
        assert(torch.sum(torch.isnan(z)) == 0)
        nr_mc_samples = z.shape[0]

        beta = self.idm.extract_samples(z, "beta")
        intercept = self.idm.extract_samples(z, "intercept")

        assert(torch.all(torch.isfinite(intercept)))
        assert(torch.all(torch.isfinite(beta)))
        assert(beta.shape[0] == nr_mc_samples and beta.shape[1] == self.data_dim)

        # ***** prior *******
        log_prior_intercept = torch.distributions.normal.Normal(loc = torch.zeros_like(intercept), scale = torch.sqrt(self.prior_variance)).log_prob(intercept)
        log_prior_beta = torch.distributions.normal.Normal(loc = torch.zeros_like(beta), scale = torch.sqrt(self.prior_variance)).log_prob(beta)
        
        # print("log_prior_intercept = ", log_prior_intercept.shape)
        # print("log_prior_beta = ", log_prior_beta.shape)
        
        assert(log_prior_beta.shape[0] == nr_mc_samples)
        assert(log_prior_beta.shape[1] == self.data_dim)

        # sum over parameter dimension
        log_prior = torch.sum(log_prior_intercept, axis = 1)
        log_prior += torch.sum(log_prior_beta, axis = 1)

        # print("log_prior = ", log_prior.shape)

        # ***** likelihood *******
        mu = self.X @ beta.t()

        # print("mu.shape = ", mu.shape)
        # print("self.y = ", self.y.shape)
        
        log_likelihood_each_observation = torch.distributions.normal.Normal(loc = mu, scale = torch.sqrt(self.likelihood_variance)).log_prob(self.y.reshape(-1,1))
        assert(log_likelihood_each_observation.shape[0] == self.X.shape[0] and log_likelihood_each_observation.shape[1] == nr_mc_samples)

        log_likelihood = torch.sum(log_likelihood_each_observation, axis = 0)

        # print("log_likelihood = ", log_likelihood.shape)
        
        assert(log_likelihood.shape[0] == nr_mc_samples)
        assert(log_likelihood.shape == log_prior.shape)
        return log_prior + log_likelihood
    



# CHECKED
class ConjugateBayesianLinearRegression(Target):

    def __init__(self, X, y):
        super().__init__()

        # Data D
        self.X =  commons.moveToDevice(X)
        self.y = commons.moveToDevice(y)
        
        self.data_dim = self.X.shape[1]
        self.n = self.X.shape[0]

        # specify dimension of each parameter
        self.idm = IndexManager(beta = self.data_dim, sigma_squared = 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = self.idm.get_pos_constraint_ids(["beta",  "sigma_squared"])

        # for evaluation calculate analytically exact marginal likelihood
        invSigma = self.getInvSigma()
        self.true_log_marginal = densities.get_log_prob_multivariate_t(commons.moveToDevice(torch.zeros(self.n)), invSigma, df = torch.tensor(1.0), y = self.y.flatten()).item()
        
        # double-check with scipy (otherwise not necessary)
        sigma = (torch.linalg.inv(invSigma)).cpu().numpy()
        mst = scipy.stats.multivariate_t(loc = numpy.zeros(self.n), shape = sigma, df = 1.0)
        y_as_matrix = self.y.cpu().numpy()
        y_as_matrix = y_as_matrix.reshape((1,-1))
        true_log_marginal_with_scipy = mst.logpdf(y_as_matrix)
        assert(sigma.shape[0] == sigma.shape[1])
        assert(sigma.shape[0] == y.shape[0])
        assert(torch.isclose(torch.tensor(self.true_log_marginal), torch.tensor(true_log_marginal_with_scipy)))

        return


    def getInvSigma(self):
        C = torch.linalg.inv(self.X.t() @ self.X + commons.moveToDevice(torch.eye(self.data_dim)))
        invSigma = commons.moveToDevice(torch.eye(self.n)) - self.X @ C @ self.X.t()
        return invSigma

    
    def log_prob(self, z):
        nr_mc_samples = z.shape[0]
        assert(z.shape[1] == self.d)
        assert(torch.sum(torch.isnan(z)) == 0)

        beta = self.idm.extract_samples(z, "beta")
        sigma_squared = self.idm.extract_samples(z, "sigma_squared")

        assert(torch.sum(torch.isnan(beta)) == 0)
        assert(torch.all(torch.isfinite(beta)))
        assert(beta.shape[1] == self.data_dim)
        assert(torch.all(torch.isfinite(sigma_squared)))
        
        # ***** prior *******
        log_prior = densities.get_log_prob_inv_gamma(sigma_squared, alpha = torch.tensor(0.5), beta = torch.tensor(0.5))
        
        # print("beta = ", beta.shape)
        # print("sigma_squared = ", sigma_squared.shape)

        log_prior_normal = torch.distributions.normal.Normal(loc = torch.zeros_like(beta), scale = torch.sqrt(sigma_squared)).log_prob(beta)
        assert(log_prior_normal.shape[1] == self.data_dim and log_prior_normal.shape[0] == nr_mc_samples)
        log_prior_normal_each_sample = torch.sum(log_prior_normal, axis = 1)

        log_prior = log_prior.squeeze()
        
        assert(log_prior.shape == log_prior_normal_each_sample.shape)
        log_prior += log_prior_normal_each_sample        

        # ***** likelihood *******
        mu = self.X @ beta.t()

        sigma_squared = sigma_squared.squeeze()
        # print("mu.shape = ", mu.shape)
        # print("sigma_squared.shape = ", sigma_squared.shape)
        # print("self.y = ", self.y.shape)
        
        assert(torch.sum(torch.isnan(mu)) == 0)
        log_likelihood_each_observation = torch.distributions.normal.Normal(loc = mu, scale = torch.sqrt(sigma_squared)).log_prob(self.y)
        assert(log_likelihood_each_observation.shape[0] == self.X.shape[0])
        log_likelihood = torch.sum(log_likelihood_each_observation, axis = 0)
        
        assert(log_likelihood.shape[0] == nr_mc_samples)
        assert(log_likelihood.shape == log_prior.shape)
        return log_prior + log_likelihood


    # used only for evaluation
    def getSamplesFromTruePosterior(self, num_samples):
        
        y = self.y.cpu().numpy()
        X = self.X.cpu().numpy()

        invU = numpy.linalg.inv(X.transpose() @ X + numpy.eye(self.data_dim))
        
        ySquaredSum = numpy.sum(numpy.square(y))
        mu = invU @ X.transpose() @ y
        mu = mu.squeeze()

        alpha = 0.5 * (1.0 + self.n)
        beta = 0.5 * (1.0 + ySquaredSum - y.transpose() @ X @ mu)
        beta = beta.item()

        nu = 1.0 + self.n
        scaleMatrix = (2.0 * beta / nu) * invU # also called shape matrix here https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html
        
        invGamma = scipy.stats.invgamma(a = alpha, scale = beta)

        mvt = scipy.stats.multivariate_t(loc = mu, shape = scaleMatrix, df=nu)

        return invGamma.rvs(size = num_samples), mvt.rvs(size = num_samples)
    

# CHECKED
class HorseshoeRegression(Target):

    # checked
    def __init__(self, X, y):
        super().__init__()
        
        # Data D
        self.X =  commons.moveToDevice(X)
        self.y = commons.moveToDevice(y)
        
        self.data_dim = self.X.shape[1]
        self.n = self.X.shape[0]
        
        # specify dimension of each parameter
        self.idm = IndexManager(lambdas = self.data_dim, tau = 1, unscaled_betas = self.data_dim, intercept = 1, sigma_squared = 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = self.idm.get_pos_constraint_ids(["lambdas", "tau", "sigma_squared"])

        self.true_log_marginal = torch.nan # unknown

        return
    
    # checked
    def log_prob(self, z):
        assert(z.shape[1] == self.d) 
        assert(torch.sum(torch.isnan(z)) == 0)

        lambdas = self.idm.extract_samples(z, "lambdas")
        tau = self.idm.extract_samples(z, "tau")
        unscaled_betas = self.idm.extract_samples(z, "unscaled_betas")
        intercept = self.idm.extract_samples(z, "intercept")

        sigma_squared = self.idm.extract_samples(z, "sigma_squared")


        # ***** prior *******
        
        lambdas_prior_log_prob = HalfCauchy(scale = torch.ones_like(lambdas)).log_prob(lambdas)
        tau_prior_log_prob = HalfCauchy(scale = torch.ones_like(tau)).log_prob(tau)
        unscaled_betas_prior_log_prob = Normal(loc = torch.zeros_like(unscaled_betas), scale = torch.ones_like(unscaled_betas)).log_prob(unscaled_betas)
        intercept_prior_log_prob = Cauchy(loc = torch.zeros_like(intercept), scale = 10.0 * torch.ones_like(intercept)).log_prob(intercept)

        lambdas_prior_log_prob = torch.sum(lambdas_prior_log_prob, axis = 1)
        tau_prior_log_prob = torch.sum(tau_prior_log_prob, axis = 1)
        unscaled_betas_prior_log_prob = torch.sum(unscaled_betas_prior_log_prob, axis = 1)
        intercept_prior_log_prob = torch.sum(intercept_prior_log_prob, axis = 1)

        sigma_squared_prior_log_prob = densities.get_log_prob_inv_gamma(sigma_squared, alpha = torch.tensor(0.5), beta = torch.tensor(0.5))
        sigma_squared_prior_log_prob = torch.sum(sigma_squared_prior_log_prob, axis = 1)

        # print("**************")
        # print("lambdas_prior_log_prob.shape = ", lambdas_prior_log_prob.shape)
        # print("tau_prior_log_prob.shape = ", tau_prior_log_prob.shape)
        # print("unscaled_betas_prior_log_prob.shape = ", unscaled_betas_prior_log_prob.shape)
        # print("**************")
        assert(sigma_squared_prior_log_prob.shape == intercept_prior_log_prob.shape)

        log_prior = lambdas_prior_log_prob + tau_prior_log_prob + unscaled_betas_prior_log_prob + intercept_prior_log_prob + sigma_squared_prior_log_prob

        # ***** likelihood *******
        scaled_betas = tau * lambdas * unscaled_betas 
        mu = self.X @ scaled_betas.t() + intercept.reshape(1, -1)

        # print("unscaled_betas.shape = ", unscaled_betas.shape)
        # print("lambdas.shape = ", lambdas.shape)
        # print("tau.shape = ", tau.shape)
        # print("scaled_betas.shape = ", scaled_betas.shape)
        # print("mu.shape = ", mu.shape)
        # print("self.y = ", self.y.shape)
        assert(mu.shape[0] == self.y.shape[0])
        # assert(mu.shape[1] >= 256)
        
        assert(torch.sum(torch.isnan(mu)) == 0)
        y_broadcasted = torch.broadcast_to(self.y, mu.shape)

        
        sigma_broadcasted = torch.sqrt(sigma_squared)
        sigma_broadcasted = sigma_broadcasted.reshape(1, -1)
        sigma_broadcasted = torch.broadcast_to(sigma_broadcasted, mu.shape)

        # print("lambdas = ", lambdas.shape)
        # print("unscaled_betas = ", unscaled_betas.shape)
        # print("mu = ", mu.shape)

        assert(torch.sum(torch.isnan(mu)) == 0)
        log_likelihood_each_observation = torch.distributions.normal.Normal(loc = mu, scale = sigma_broadcasted).log_prob(y_broadcasted)
        assert(log_likelihood_each_observation.shape[0] == self.X.shape[0]) 

        log_likelihood = torch.sum(log_likelihood_each_observation, axis = 0)

        # print("log_likelihood = ", log_likelihood)
        # print("log_prior = ", log_prior)
        # assert(False)
        
        assert(log_likelihood.shape[0] == z.shape[0])
        assert(log_likelihood.shape == log_prior.shape)
        return log_prior + log_likelihood
    

# not yet fully checked
# The Bayesian Lasso as used in "Monte Carlo Approximation of Bayes Factors via Mixing With Surrogate Distributions", JASA, 2020
class BayesianLasso(Target):

    # checked
    def __init__(self, X, y, lambda_hyper_param = 1.0):
        super().__init__()
        assert(lambda_hyper_param >= 0.1 and lambda_hyper_param <= 100.0)
        
        # Data D
        self.X =  commons.moveToDevice(X)
        self.y = commons.moveToDevice(y)
        
        self.data_dim = self.X.shape[1]
        self.n = self.X.shape[0]
        
        # specify dimension of each parameter
        self.idm = IndexManager(all_tau_squared = self.data_dim, sigma_squared = 1, beta = self.data_dim, intercept = 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = self.idm.get_pos_constraint_ids(["all_tau_squared", "sigma_squared"])

        self.true_log_marginal = torch.nan # unknown
        
        
        # register non-parameters (this is useful for example to ensure that their are also moved to the GPU if you call model.cuda() etc.)
        self.register_buffer('lambda_hyper_param', torch.tensor([lambda_hyper_param]))  
        
        return
    
    # checked
    def log_prob(self, z):
        nr_mc_samples = z.shape[0]
        assert(z.shape[1] == self.d) 
        assert(torch.sum(torch.isnan(z)) == 0)

        beta = self.idm.extract_samples(z, "beta")
        intercept = self.idm.extract_samples(z, "intercept")

        all_tau_squared = self.idm.extract_samples(z, "all_tau_squared")
        sigma_squared = self.idm.extract_samples(z, "sigma_squared")

        # ***** prior *******
        diagonal_scales = torch.sqrt(all_tau_squared * sigma_squared) + EPSILON
        assert(diagonal_scales.shape == beta.shape)

        betas_prior_log_prob = Normal(loc = torch.zeros_like(beta), scale = diagonal_scales).log_prob(beta)
        betas_prior_log_prob = torch.sum(betas_prior_log_prob, axis = 1)
        
        intercept_prior_log_prob = Cauchy(loc = torch.zeros_like(intercept), scale = 10.0 * torch.ones_like(intercept)).log_prob(intercept)
        intercept_prior_log_prob = torch.sum(intercept_prior_log_prob, axis = 1)

        sigma_squared_prior_log_prob = densities.get_log_prob_inv_gamma(sigma_squared, alpha = torch.tensor(0.5), beta = torch.tensor(0.5))
        sigma_squared_prior_log_prob = torch.sum(sigma_squared_prior_log_prob, axis = 1)
        
        tau_squared_prior_log_prob = Exponential(rate = (self.lambda_hyper_param ** 2) / 2.0).log_prob(all_tau_squared)
        tau_squared_prior_log_prob = torch.sum(tau_squared_prior_log_prob, axis = 1)
        
        log_prior = betas_prior_log_prob + intercept_prior_log_prob + sigma_squared_prior_log_prob + tau_squared_prior_log_prob 

        assert(torch.sum(torch.isnan(self.X)) == 0)
        assert(torch.sum(torch.isnan(beta)) == 0)

        assert(torch.all(torch.isfinite(sigma_squared)))
        assert(torch.all(torch.isfinite(beta)))

        mu = self.X @ beta.t() + intercept.reshape(1, -1)

        y_broadcasted = torch.broadcast_to(self.y, mu.shape)

        sigma_broadcasted = torch.sqrt(sigma_squared)
        sigma_broadcasted = sigma_broadcasted.reshape(1, -1)
        sigma_broadcasted = torch.broadcast_to(sigma_broadcasted, mu.shape)

        assert(torch.sum(torch.isnan(mu)) == 0)
        log_likelihood_each_observation = torch.distributions.normal.Normal(loc = mu, scale = sigma_broadcasted).log_prob(y_broadcasted)
        assert(log_likelihood_each_observation.shape[0] == self.X.shape[0]) 

        log_likelihood = torch.sum(log_likelihood_each_observation, axis = 0)

        assert(log_likelihood.shape[0] == nr_mc_samples)
        assert(log_likelihood.shape == log_prior.shape)
        return log_prior + log_likelihood


# not yet fully checked
class HorseshoePriorLogisticRegression(Target):

    # checked
    def __init__(self, X, y):
        super().__init__()
        
        # Data D
        self.X =  commons.moveToDevice(X)
        self.y = commons.moveToDevice(y)
        
        self.data_dim = self.X.shape[1]
        self.n = self.X.shape[0]
        
        # specify dimension of each parameter
        self.idm = IndexManager(lambdas = self.data_dim, tau = 1, unscaled_betas = self.data_dim, intercept = 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = self.idm.get_pos_constraint_ids(["lambdas", "tau"])

        self.true_log_marginal = torch.nan # unknown

        return
    

    # checked
    def log_prob(self, z):
        assert(z.shape[1] == self.d) 
        assert(torch.sum(torch.isnan(z)) == 0)

        lambdas = self.idm.extract_samples(z, "lambdas")
        tau = self.idm.extract_samples(z, "tau")
        unscaled_betas = self.idm.extract_samples(z, "unscaled_betas")
        intercept = self.idm.extract_samples(z, "intercept")
        
        # ***** prior *******
        
        lambdas_prior_log_prob = HalfCauchy(scale = torch.ones_like(lambdas)).log_prob(lambdas)
        tau_prior_log_prob = HalfCauchy(scale = torch.ones_like(tau)).log_prob(tau)
        unscaled_betas_prior_log_prob = Normal(loc = torch.zeros_like(unscaled_betas), scale = torch.ones_like(unscaled_betas)).log_prob(unscaled_betas)
        intercept_prior_log_prob = Cauchy(loc = torch.zeros_like(intercept), scale = 10.0 * torch.ones_like(intercept)).log_prob(intercept)

        lambdas_prior_log_prob = torch.sum(lambdas_prior_log_prob, axis = 1)
        tau_prior_log_prob = torch.sum(tau_prior_log_prob, axis = 1)
        unscaled_betas_prior_log_prob = torch.sum(unscaled_betas_prior_log_prob, axis = 1)
        intercept_prior_log_prob = torch.sum(intercept_prior_log_prob, axis = 1)
        
        # print("**************")
        # print("lambdas_prior_log_prob.shape = ", lambdas_prior_log_prob.shape)
        # print("tau_prior_log_prob.shape = ", tau_prior_log_prob.shape)
        # print("unscaled_betas_prior_log_prob.shape = ", unscaled_betas_prior_log_prob.shape)
        # print("**************")

        log_prior = lambdas_prior_log_prob + tau_prior_log_prob + unscaled_betas_prior_log_prob + intercept_prior_log_prob

        # ***** likelihood *******
        scaled_betas = tau * lambdas * unscaled_betas 
        mu = self.X @ scaled_betas.t() + intercept.reshape(1, -1)

        # print("unscaled_betas.shape = ", unscaled_betas.shape)
        # print("lambdas.shape = ", lambdas.shape)
        # print("tau.shape = ", tau.shape)
        # print("scaled_betas.shape = ", scaled_betas.shape)
        # print("mu.shape = ", mu.shape)
        # print("self.y = ", self.y.shape)
        assert(mu.shape[0] == self.y.shape[0])
        # assert(mu.shape[1] >= 256)
        
        assert(torch.sum(torch.isnan(mu)) == 0)
        y_broadcasted = torch.broadcast_to(self.y, mu.shape)
        log_likelihood_each_observation = Bernoulli(logits=mu).log_prob(y_broadcasted)

        assert(log_likelihood_each_observation.shape[0] == self.X.shape[0])
        # assert(log_likelihood_each_observation.shape[1] >= 256)

        log_likelihood = torch.sum(log_likelihood_each_observation, axis = 0)

        assert(log_likelihood.shape[0] == z.shape[0])
        assert(log_likelihood.shape == log_prior.shape)
        return log_prior + log_likelihood
    


# **************************************
# ************************ below are some typical distributions that are used for benchmarking (but these are not Bayesian models, therefore there is not X or y in the constructor) ********
# **************************************

# CHECKED
# Funnel Distribution as in "Slice sampling", Neal 2003
class Funnel(Target):
    
    def __init__(self,dim = 10, scale = 3.0):
        super().__init__()

        # register non-parameters (this is useful for example to ensure that their are also moved to the GPU if you call model.cuda() etc.)
        self.register_buffer('scale', torch.tensor([scale]))  
        
        # specify dimension of each parameter
        self.idm = IndexManager(theta_one = 1, theta_remaining = dim - 1)
        self.d = self.idm.d # total number of parameters

        # specify parameters which domain is positive
        self.pos_constraint_ids = None

        self.true_log_marginal = 0.0
        return

    def log_prob(self, z):
        nr_mc_samples = z.shape[0]
        theta_one = self.idm.extract_samples(z, "theta_one")
        theta_remaining = self.idm.extract_samples(z, "theta_remaining")

        log_prob_theta_one_each_sample = torch.distributions.normal.Normal(0,self.scale).log_prob(theta_one)

        std_other = torch.sqrt(torch.exp(log_prob_theta_one_each_sample)) + EPSILON
        assert((std_other > 0.0).all())

        std_other = torch.broadcast_to(std_other, theta_remaining.shape)
        log_prob_other = torch.distributions.normal.Normal(0.0, std_other).log_prob(theta_remaining)

        each_sample_cond_log_prob = torch.sum(log_prob_other, 1)
        assert(each_sample_cond_log_prob.shape[0] == nr_mc_samples)
        
        log_prob_theta_one_each_sample = log_prob_theta_one_each_sample.squeeze()
        assert(log_prob_theta_one_each_sample.shape == each_sample_cond_log_prob.shape)
        log_prob_each_sample = log_prob_theta_one_each_sample + each_sample_cond_log_prob
        
        assert(z.shape[0] == log_prob_each_sample.shape[0])
        return log_prob_each_sample


    # used only for evaluation
    def getSamplesFromTruePosterior(self, num_samples):
        
        v = scipy.stats.norm(loc = 0.0, scale = self.scale).rvs(num_samples)
        std_other = numpy.sqrt(numpy.exp(v)) + EPSILON
        assert((std_other > 0.0).all())

        std_other = numpy.broadcast_to(std_other, (self.d - 1, num_samples)).transpose()

        remaining_samples = scipy.stats.norm(loc = numpy.zeros_like(std_other), scale = std_other).rvs(size = (num_samples, self.d - 1))

        z = numpy.hstack((v.reshape(-1,1), remaining_samples))
        return z


# CHECKED
# Multivariate Normal Mixture 
class MultivariateNormalMixture(Target):

    FACTOR_FOR_MEAN = 6.0

    def getAdjustedMeanFac(dim, meanFac =  FACTOR_FOR_MEAN):
        return meanFac * math.sqrt(1.0 / dim)
    
    def __init__(self,dim = 10, K = 3, meanFac =  FACTOR_FOR_MEAN):
        # assert(K == 3 or K == 2)
        # assert(meanFac >= 0.1 and meanFac <= 10.0)
        assert(K == 3 and meanFac == 6.0)
        assert(dim >= 10)
        
        # specify dimension of each parameter
        self.idm = IndexManager(theta = dim)
        self.d = self.idm.d # total number of parameters

        meanFac = MultivariateNormalMixture.getAdjustedMeanFac(dim, meanFac)
        print("dim adjusted meanFac = ", meanFac)
        
        super().__init__()

        self.K = K
        self.mu3 = torch.zeros(dim)
        
        self.mu1 = meanFac * torch.ones(dim)
        self.mu2 = -meanFac * torch.ones(dim)
        self.cov = torch.eye(dim)
        
        uniform_dist = torch.ones(K) / K

        self.mu1 = commons.moveToDevice(self.mu1)
        self.mu2 = commons.moveToDevice(self.mu2)
        self.mu3 = commons.moveToDevice(self.mu3)
        self.cov = commons.moveToDevice(self.cov)
        uniform_dist = commons.moveToDevice(uniform_dist)

        mixture_weights = dist.Categorical(uniform_dist)
        
        if K == 2:
            components = dist.MultivariateNormal(torch.stack([self.mu1,self.mu2]), torch.stack([self.cov,self.cov]))
        elif K == 3:
            components = dist.MultivariateNormal(torch.stack([self.mu1,self.mu2,self.mu3]), torch.stack([self.cov,self.cov,self.cov]))
        else:
            assert(False)
        
        self.gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mixture_weights, components)
        
        self.true_log_marginal = 0.0

        # specify parameters which domain is positive
        self.pos_constraint_ids = None
        return
    
    
    def log_prob(self, z):
        
        log_prob_each_sample = self.gmm.log_prob(z)

        assert(log_prob_each_sample.shape[0] == z.shape[0])
        return log_prob_each_sample
    
    # used only for evaluation
    def getSamplesFromTruePosterior(self, num_samples):
        
        all_samples = numpy.zeros((num_samples, self.d))

        all_c = numpy.random.choice(self.K, num_samples)
        
        mu1 = self.mu1.cpu().numpy()
        mu2 = self.mu2.cpu().numpy()
        mu3 = self.mu3.cpu().numpy()
        cov = self.cov.cpu().numpy()

        if self.K == 2:
            all_mvn_dists = [scipy.stats.multivariate_normal(mean = mu1, cov = cov), scipy.stats.multivariate_normal(mean =mu2, cov = cov)]
        elif self.K == 3:
            all_mvn_dists = [scipy.stats.multivariate_normal(mean = mu1, cov = cov), scipy.stats.multivariate_normal(mean = mu2, cov = cov), scipy.stats.multivariate_normal(mean = mu3, cov = cov)]
        else:
            assert(False)

        filled_unit_id = 0
        for cluster_id in range(self.K):
            nr_samples_this_cluster = numpy.sum(all_c == cluster_id)
            all_samples[filled_unit_id : (filled_unit_id + nr_samples_this_cluster), :] = all_mvn_dists[cluster_id].rvs(size = nr_samples_this_cluster)
            filled_unit_id += nr_samples_this_cluster
            
        assert(filled_unit_id == num_samples)
        
        return all_samples


# CHECKED
class MultivariateStudentT(Target):
   
    def __init__(self, dim, df = 1.0):
        super().__init__()

        # specify dimension of each parameter
        self.idm = IndexManager(theta = dim)
        self.d = self.idm.d # total number of parameters

        var = 1.0
        
        self.df = torch.tensor(df)
        self.dim = torch.tensor(dim)
        self.loc = 2.0 * torch.ones(dim)
        self.cov = var * (torch.ones((dim,dim))*0.8 + 0.2*torch.eye(dim))
        print("variance of MultivariateStudentT = ", var)

        self.L = torch.linalg.cholesky(torch.inverse(self.cov))

        self.fixedTerms = torch.lgamma((self.df + self.dim)/2.0) \
                            - torch.lgamma(self.df/2.0) - (self.dim/2.0) * torch.log(self.df) \
                            - (self.dim/2.0) * torch.log(torch.tensor(torch.pi)) - 0.5 * torch.logdet(self.cov)
        
        self.df = commons.moveToDevice(self.df)
        self.dim = commons.moveToDevice(self.dim)
        self.loc = commons.moveToDevice(self.loc)
        self.cov = commons.moveToDevice(self.cov)
        self.L = commons.moveToDevice(self.L)
        self.fixedTerms = commons.moveToDevice(self.fixedTerms)
        
        self.true_log_marginal = 0.0

        # specify parameters which domain is positive
        self.pos_constraint_ids = None

        return
    
    def log_prob(self, z):
        # assert(z.shape == self.loc.shape)

        zL = (z - self.loc) @ self.L

        # print("z.shape = ", z.shape)
        # print("self.loc.shape = ", self.loc.shape)
        # print("zL.shape = ", zL.shape)
        # print("(z - self.loc) = ", (z - self.loc).shape)
        # assert(False)

        log_prob_each_sample = self.fixedTerms - ((self.df + self.dim)/2.0) * torch.log(1.0 + (1.0/self.df)*torch.sum(torch.square(zL),1))

        assert(z.shape[0] == log_prob_each_sample.shape[0])

        return log_prob_each_sample

    # used only for evaluation
    def getSamplesFromTruePosterior(self, num_samples):
        loc = self.loc.cpu().numpy()
        cov = self.cov.cpu().numpy()
        df = self.df.cpu().numpy()
        
        mvt = scipy.stats.multivariate_t(loc = loc, shape = cov, df=df)
        return mvt.rvs(size = num_samples)


    








    

    
