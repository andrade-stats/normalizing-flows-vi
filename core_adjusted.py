import torch
from normflows.distributions import Target
import normflows as nf
import torch.nn as nn
import numpy as np
import new_flows
import commons


def filter_illegal_values_from_samples(z, log_q):
    failed = False
    invalid_value_found = False

    if torch.any(torch.isnan(z)) or torch.any(torch.isnan(log_q)):
        total_nr_samples = z.shape[0]

        # torch.set_printoptions(profile="full")
        # print("z = ")
        # print(z)
        # print("log_q = ")
        # print(log_q)

        assert(z.shape[0] >= 256)
        nan_sample_ids_z = torch.isnan(torch.sum(z, axis = 1))
        nan_sample_ids_log_q = torch.isnan(log_q)
        nan_ids = torch.logical_or(nan_sample_ids_z, nan_sample_ids_log_q)

        z = z[~nan_ids, :]
        log_q = log_q[~nan_ids]

        print("total number of samples = ", total_nr_samples)
        print("number of samples with nan entries = ", torch.sum(nan_ids).item())
        
        if torch.sum(nan_ids).item() == total_nr_samples:
            failed = True
        
        # torch.set_printoptions(profile="full")
        # print("z = ")
        # print(z)
        # print("log_q = ")
        # print(log_q)
        invalid_value_found = True
    
    if (~torch.all(torch.isfinite(z)) or ~torch.all(torch.isfinite(log_q))):
        total_nr_samples = z.shape[0]
        finite_ids_z = torch.all(torch.isfinite(z), axis = 1)
        finite_ids_log_q = torch.isfinite(log_q)
        assert(finite_ids_z.shape[0] == finite_ids_log_q.shape[0])
        finite_ids = torch.logical_and(finite_ids_z, finite_ids_log_q)
        print("number of infinite samples = ", ( total_nr_samples - torch.sum(finite_ids).item()))
        # assert(torch.sum(finite_ids).item() > 0)
        if torch.sum(finite_ids).item() == 0:
            failed = True

        z = z[finite_ids, :]
        log_q = log_q[finite_ids]
        invalid_value_found = True
    
    return z, log_q, invalid_value_found, failed


def reverse_kld(nfm, num_samples=1, beta=1.0, show_details = False):
    """Adjusted from original implementation in core.py of normflows packages

    Args:
        num_samples: Number of samples to draw from base distribution
        beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
        
    Returns:
        Estimate of the reverse KL divergence averaged over latent samples
    """
    
    z, log_q = nfm.sample(num_samples)
    z, log_q, invalid_value_found, failed = filter_illegal_values_from_samples(z, log_q)

    log_p = nfm.p.log_prob(z)
    
    
    not_nan_ids = torch.logical_and( torch.logical_not(torch.isnan(log_q)) , torch.logical_not(torch.isnan(log_p)))
    if torch.sum(not_nan_ids) == 0:
        print("** WARNING too many nans !!")
        print("not nan ratio = ", torch.sum(not_nan_ids) / not_nan_ids.shape[0])
        print("nan in q = ", torch.sum(torch.isnan(log_q)))
        print("nan in p = ", torch.sum(torch.isnan(log_p)))
    
    if show_details:
        infinite_count_q = torch.sum(torch.logical_not(torch.isfinite(log_q)))
        infinite_count_p = torch.sum(torch.logical_not(torch.isfinite(log_p)))
        print("---")
        print("infinite_count_q = ", infinite_count_q)
        print("infinite_count_p = ", infinite_count_p)
        assert(False)
    
    log_p = log_p[not_nan_ids]
    log_q = log_q[not_nan_ids]
    kld_with_score = log_q - beta * log_p

    return kld_with_score, log_q.detach(), log_p.detach()



def reverse_kld_without_score_debug(mixture_nfm, num_samples=1, beta = 1.0):
    assert(len(mixture_nfm.all_flows) == 1)
    nfm = mixture_nfm.all_flows[0]

    # assert(mixture_nfm.number_of_flows == 64)
    z_stats_median = torch.zeros(mixture_nfm.number_of_flows + 1)
    z_stats_high = torch.zeros(mixture_nfm.number_of_flows + 1)
    z_stats_higher = torch.zeros(mixture_nfm.number_of_flows + 1)
    z_stats_max = torch.zeros(mixture_nfm.number_of_flows + 1)
    masked_affine_flow_id = 0

    z, log_q_ = nfm.q0(num_samples)
    
    z_abs = torch.abs(z.detach())
    z_stats_median[masked_affine_flow_id] = z_abs.nanquantile(0.5)
    z_stats_high[masked_affine_flow_id] = z_abs.nanquantile(0.75)
    z_stats_higher[masked_affine_flow_id] = z_abs.nanquantile(0.90)
    z_stats_max[masked_affine_flow_id] = z_abs.nanquantile(1.0)
    masked_affine_flow_id += 1

    log_q = torch.zeros_like(log_q_)
    log_q += log_q_
    for flow in nfm.flows:
        z, log_det = flow(z)
        log_q -= log_det

        if (type(flow) is nf.flows.MaskedAffineFlow) or (type(flow) is new_flows.MaskedAffineFlowThresholded) or (type(flow) is new_flows.MaskedAffineFlowSquashedSigmoid) or (type(flow) is new_flows.MaskedAffineFlowSoftClamp):
            z_abs = torch.abs(z.detach())
            z_stats_median[masked_affine_flow_id] = z_abs.nanquantile(0.5)
            z_stats_high[masked_affine_flow_id] = z_abs.nanquantile(0.75)
            z_stats_higher[masked_affine_flow_id] = z_abs.nanquantile(0.90)
            z_stats_max[masked_affine_flow_id] = z_abs.nanquantile(1.0)
            masked_affine_flow_id += 1

    if (type(flow) is nf.flows.MaskedAffineFlow) or (type(flow) is new_flows.MaskedAffineFlowThresholded) or (type(flow) is new_flows.MaskedAffineFlowSquashedSigmoid) or (type(flow) is new_flows.MaskedAffineFlowSoftClamp):
        assert(masked_affine_flow_id == mixture_nfm.number_of_flows + 1)
    
    z, log_q, invalid_value_found, failed = filter_illegal_values_from_samples(z, log_q)
    
    if z.shape[0] == 0:
        empty_tensor = torch.tensor([])
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor

    log_p = nfm.p.log_prob(z)

    z_prime = z.detach()
    score_term = nfm.log_prob(z_prime)

    kld_with_score = log_q - beta * log_p
    kld_without_score = kld_with_score - score_term

    nan_ids = torch.logical_or(torch.isnan(log_p), torch.isnan(kld_without_score))
    z_prime = z_prime[~nan_ids, :]
    log_p = log_p[~nan_ids]
    log_q = log_q[~nan_ids]
    kld_with_score = kld_with_score[~nan_ids]
    kld_without_score = kld_without_score[~nan_ids]
    
    assert(not torch.any(torch.isnan(log_p)))
    assert(not torch.any(torch.isnan(kld_with_score)))
    assert(not torch.any(torch.isnan(kld_without_score)))
    assert(not torch.any(torch.isnan(log_q)))
    assert(not torch.any(torch.isnan(z_prime)))

    return kld_without_score, kld_with_score, z_stats_median, z_stats_high, z_stats_higher, z_stats_max


def reverse_kld_without_score(nfm, num_samples=1, beta=1.0):

    z, log_q = nfm.sample(num_samples)
    z, log_q, invalid_value_found, failed = filter_illegal_values_from_samples(z, log_q)
    
    if z.shape[0] == 0:
        empty_tensor = torch.tensor([])
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor

    log_p = nfm.p.log_prob(z)

    z_prime = z.detach()
    score_term = nfm.log_prob(z_prime)

    kld_with_score = log_q - beta * log_p
    kld_without_score = kld_with_score - score_term

    nan_ids = torch.logical_or(torch.isnan(log_p), torch.isnan(kld_without_score))
    z_prime = z_prime[~nan_ids, :]
    log_p = log_p[~nan_ids]
    log_q = log_q[~nan_ids]
    kld_with_score = kld_with_score[~nan_ids]
    kld_without_score = kld_without_score[~nan_ids]
    
    assert(not torch.any(torch.isnan(log_p)))
    assert(not torch.any(torch.isnan(kld_with_score)))
    assert(not torch.any(torch.isnan(kld_without_score)))
    assert(not torch.any(torch.isnan(log_q)))
    assert(not torch.any(torch.isnan(z_prime)))

    return kld_without_score, kld_with_score, log_q.detach(), log_p.detach(), z_prime


class DiagGaussian(nf.distributions.BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, initial_loc = None, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            if initial_loc is None:
                self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            else:
                assert(initial_loc.shape[0] == 1 and initial_loc.shape[1] == shape[0])
                self.loc = nn.Parameter(initial_loc)
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            assert(initial_loc is not None)
            self.register_buffer("loc", initial_loc)
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class DiagStudentT(nf.distributions.BaseDistribution):
    """
    Standard StudentT with trainable degrees of freedom in each dimension
    """

    # if trainable=False, then only the degree-of-freedom nu will be learnt
    def __init__(self, shape, initial_loc = None, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        
        initial_log_deg = torch.log(torch.ones(self.d) * (30.0 - 1.0)) # recommended initial value in "Tails of Lipschitz Triangular Flows", ICML, 2020
        initial_log_deg = commons.moveToDevice(initial_log_deg)
        self.log_deg_freedom = nn.Parameter(initial_log_deg) # note that each dimension can learn a different degree of freedom as suggested in "Fat-Tailed Variational Inference with Anisotropic Tail Adaptive Flows", ICML, 2022

        initial_log_scales = commons.moveToDevice(torch.zeros(self.d))

        if trainable:
            if initial_loc is None:
                self.loc = nn.Parameter(commons.moveToDevice(torch.zeros(self.d)))
            else:
                assert(initial_loc.shape[0] == 1 and initial_loc.shape[1] == shape[0])
                self.loc = nn.Parameter(torch.squeeze(initial_loc))
            self.log_scale = nn.Parameter(initial_log_scales)
        else:
            assert(initial_loc is not None)
            self.register_buffer("loc", torch.squeeze(initial_loc))
            self.register_buffer("log_scale", initial_log_scales)


    def forward(self, num_samples=1):
        
        nu = 1.0 + torch.exp(self.log_deg_freedom)
        studentT = torch.distributions.studentT.StudentT(df = nu, loc = self.loc, scale = torch.exp(self.log_scale))
        
        z = studentT.rsample(torch.Size([num_samples]))
        log_p = studentT.log_prob(z)
        log_p = torch.sum(log_p, dim = 1)

        assert(z.shape[0] == num_samples) # batch size
        assert(z.shape[1] >= 10)

        return z, log_p


    def log_prob(self, z):
        nu = 1.0 + torch.exp(self.log_deg_freedom)
        studentT = torch.distributions.studentT.StudentT(df = nu, loc = self.loc, scale = torch.exp(self.log_scale))
        
        nan_sample_ids_z = torch.isnan(torch.sum(z, axis = 1))
        
        if torch.any(nan_sample_ids_z):
            z_tmp = torch.clone(z)
            z_tmp[nan_sample_ids_z, :] = -1.0
            log_p = studentT.log_prob(z_tmp)
            log_p = torch.sum(log_p, dim = 1)
            log_p[nan_sample_ids_z] = torch.nan
        else:
            log_p = studentT.log_prob(z)
            log_p = torch.sum(log_p, dim = 1)
        
        return log_p



