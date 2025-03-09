import torch
import core_adjusted
import commons



NUM_SAMPLES = 20000

# estimates the marginal likelihood using importance sampling with q(=normalizing flow) as the proposal distribution
def importance_sampling(nfm, num_samples = NUM_SAMPLES):

    z,log_q = nfm.sample(num_samples = num_samples)
    z,log_q, _, _ = core_adjusted.filter_illegal_values_from_samples(z, log_q)
    log_p = nfm.p.log_prob(z)

    log_mC = torch.logsumexp(log_p - log_q, dim = 0)
    log_is_estimate = log_mC - torch.log(torch.tensor(num_samples))

    return log_is_estimate.item()


def elbo_estimate(nfm, num_samples = NUM_SAMPLES):
    negative_elbo_samples, _, _ = core_adjusted.reverse_kld(nfm, num_samples)
    assert(negative_elbo_samples.shape[0] == num_samples)

    return - torch.mean(negative_elbo_samples).item()



# get posterior samples using the approximation(=normalizing flow)
def get_posterior_samples(nfm, num_samples = NUM_SAMPLES):
    with torch.no_grad():
        z,log_q = nfm.sample(num_samples)

    z,log_q,_, _ = core_adjusted.filter_illegal_values_from_samples(z, log_q)
    
    target = nfm.p
    assert(z.shape[0] >= 256) # Monte Carlo Samples
    assert(z.shape[1] == target.d)

    param_to_samples = {}

    for param in target.idm.get_all_param_names():
        samples = target.idm.extract_samples(z, param)
        assert(torch.all(torch.isfinite(samples)))
        param_to_samples[param] = samples.cpu().numpy()

    log_q = log_q.cpu().numpy()
    log_p = target.log_prob(z).cpu().numpy()

    return param_to_samples, log_p, log_q


def getRepeatedEstimates(nfm, type, num_samples = NUM_SAMPLES):
    
    with torch.no_grad():
        all_estimates = torch.zeros(commons.REPETITIONS_FOR_MC_ERROR)
        for i in range(commons.REPETITIONS_FOR_MC_ERROR):
            if type == "ELBO":
                reverse_kld, _, _ = core_adjusted.reverse_kld(nfm, num_samples=num_samples)
                all_estimates[i] = - torch.mean(reverse_kld)
            elif type == "IS":
                all_estimates[i] = importance_sampling(nfm, num_samples)
            else:
                assert(False)

    return all_estimates.numpy()
