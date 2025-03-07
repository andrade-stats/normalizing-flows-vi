
import numpy
import torch
import commons
import normflows as nf
from tqdm import tqdm
import core_adjusted
from normflows.flows.affine.coupling import AffineConstFlow
import analysis
import new_flows

def getNormalizingFlow(target, flow_type, number_of_flows, initial_loc_base = None, use_student_base = True, use_LOFT = True, hidden_layer_size_spec = 100):
    
    # these hyper-parameters work often good in practise, see also "Stabilizing training of affine coupling layers for high-dimensional variational inference", 2024
    LOFT_THRESHOLD_VALUE = 100.0
    REAL_NVP_SCALING_FACTOR_BOUND_VALUE = 0.1
    REAL_NVP_SCALING_FACTOR_BOUND_TYPE = "asymmetric"

    latent_size = target.d
    binary_mask = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    binary_mask = commons.moveToDevice(binary_mask)
    flows = []

    if hidden_layer_size_spec == "double":
        hidden_layer_size = 2 * latent_size
    elif hidden_layer_size_spec is not None:
        hidden_layer_size = int(hidden_layer_size_spec)
        assert(hidden_layer_size > 10 and hidden_layer_size < 1000)
    

    if flow_type == "RealNVP":
        trainable_base = False
    else:
        trainable_base = True
    

    if flow_type != "GaussianOnly":
        
        for i in range(number_of_flows):
            if flow_type == "RealNVP":

                scale_nn = nf.nets.MLP([latent_size, hidden_layer_size, latent_size], init_zeros=True)
                translation_nn = nf.nets.MLP([latent_size, hidden_layer_size, latent_size], init_zeros=True)

                if REAL_NVP_SCALING_FACTOR_BOUND_TYPE is not None:
                    if i % 2 == 0:
                        flows += [new_flows.MaskedAffineFlowThresholded(binary_mask, translation_nn, scale_nn, REAL_NVP_SCALING_FACTOR_BOUND_VALUE, REAL_NVP_SCALING_FACTOR_BOUND_TYPE)]
                    else:
                        flows += [new_flows.MaskedAffineFlowThresholded(1 - binary_mask, translation_nn, scale_nn, REAL_NVP_SCALING_FACTOR_BOUND_VALUE, REAL_NVP_SCALING_FACTOR_BOUND_TYPE)]
                else:
                    if i % 2 == 0:
                        flows += [nf.flows.MaskedAffineFlow(binary_mask, translation_nn, scale_nn)]
                    else:
                        flows += [nf.flows.MaskedAffineFlow(1 - binary_mask, translation_nn, scale_nn)]
                
                
                if i == number_of_flows - 1:
                    # add after last RealNVP layer
                    if use_LOFT:
                        flows += [new_flows.TrainableLOFTLayer(latent_size, LOFT_THRESHOLD_VALUE, train_t = False)]
                
            elif flow_type == "MAF":
                flows += [nf.flows.MaskedAffineAutoregressive(latent_size,2*latent_size)]
                
            elif flow_type == "Planar":
                flows += [nf.flows.Planar((latent_size,), act = "leaky_relu")]
                
            elif flow_type == "NeuralSpline":
                flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, 1, hidden_layer_size, reverse_mask = (i % 2 == 0))]
                
            else:
                assert(False)
    else:
        assert(number_of_flows == 0)
        assert(not use_student_base)
        assert(not use_LOFT)
        

    if not trainable_base:
        # use a final affine transformation, instead of trainable location and scale of base distribution
        flows += [AffineConstFlow(latent_size)]
    
    if target.pos_constraint_ids is not None:
        flows += [new_flows.PositiveConstraintLayer(target.pos_constraint_ids, target.d)]
    
    if use_student_base:
        q0 = core_adjusted.DiagStudentT(target.d, initial_loc = initial_loc_base, trainable = trainable_base) 
    else:
        q0 = core_adjusted.DiagGaussian(target.d, initial_loc = initial_loc_base, trainable = trainable_base) 

    target = commons.moveToDevice(target)

    # construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
    nfm = commons.moveToDevice(nfm)

    return nfm


    
class FlowsMixture(torch.nn.Module):

    def __init__(self, target, nr_mixture_components, flow_type, number_of_flows, learn_mixture_weights = False, initial_loc_spec = "zeros", use_student_base = True, use_LOFT = True, hidden_layer_size_spec = 100):
        super().__init__()

        self.d = target.d
        self.p = commons.moveToDevice(target)  # register the target distribution as "p" (naming as in normflows package)
        self.number_of_flows = number_of_flows

        uniform_dist = torch.ones(nr_mixture_components) / nr_mixture_components
        self.mixture_weights = commons.moveToDevice(uniform_dist)

        if learn_mixture_weights:
            self.mixture_weights = torch.nn.Parameter(self.mixture_weights)
        
        all_flows = []
        for k in range(nr_mixture_components):

            if initial_loc_spec == "random_large":
                initial_loc_for_flow = torch.rand(size = (1,target.d)) * 20.0 - 10.0
            elif initial_loc_spec == "random_small":
                initial_loc_for_flow = torch.rand(size = (1,target.d)) * 2.0 - 1.0
            elif initial_loc_spec == "zeros":
                assert(nr_mixture_components == 1)
                initial_loc_for_flow = torch.zeros(size = (1,target.d))
            else:
                assert(False)
            
            initial_loc_for_flow = commons.moveToDevice(initial_loc_for_flow)

            nfm = getNormalizingFlow(target, flow_type = flow_type, number_of_flows = number_of_flows, initial_loc_base = initial_loc_for_flow, use_student_base = use_student_base, use_LOFT = use_LOFT, hidden_layer_size_spec = hidden_layer_size_spec)
            all_flows.append(nfm)

        self.all_flows = torch.nn.ModuleList(all_flows)  # register all flows (this makes all parameters learnable)

        return
    

    # checked
    def sample(self, num_samples):

        multinomial = torch.distributions.multinomial.Multinomial(num_samples, probs= self.mixture_weights)
        freq_each_comp = multinomial.sample()

        assert(len(self.all_flows) == self.mixture_weights.shape[0])

        log_comp_weights = torch.log(self.mixture_weights)

        all_z = []
        all_log_q = []
        for k in range(len(self.all_flows)):
            z, log_q_comp_k = self.all_flows[k].sample(int(freq_each_comp[k]))
            all_z.append(z)

            log_q_each_comp = []
            log_q_each_comp.append(log_q_comp_k + log_comp_weights[k])
            for j in range(len(self.all_flows)):
                if j != k:
                    log_q_comp_j = self.all_flows[j].log_prob(z)
                    log_q_each_comp.append(log_q_comp_j + log_comp_weights[j])
            
            log_q_each_comp = torch.stack(log_q_each_comp)
            log_q = torch.logsumexp(log_q_each_comp, dim = 0)
            all_log_q.append(log_q)

        all_z = torch.vstack(all_z)
        all_log_q = torch.hstack(all_log_q)

        assert(all_z.shape[0] == num_samples and all_z.shape[1] == self.d)
        assert(all_log_q.shape[0] == num_samples)

        return all_z, all_log_q
    
    # checked
    def log_prob(self, z):

        log_comp_weights = torch.log(self.mixture_weights)

        log_q_each_comp = []

        for k in range(len(self.all_flows)):
            log_q_comp_k = self.all_flows[k].log_prob(z)
            log_q_each_comp.append(log_q_comp_k + log_comp_weights[k]) 
            
        log_q_each_comp = torch.stack(log_q_each_comp)
        all_log_q = torch.logsumexp(log_q_each_comp, dim = 0)
        
        assert(all_log_q.shape[0] == z.shape[0])

        return all_log_q
    



    




def train(nfm, max_iter, learning_rate = 10 ** (-4), divergence = "reverse_kld_without_score", num_mc_samples = 2 ** 8, annealing = False, anneal_iter = None, record_stats = True):

    print("max_iter = ", max_iter)
    print("annealing = ", annealing)
    print("divergence = ", divergence)

    if max_iter < 50:
        show_iter = 1
    elif max_iter <= 100:
        show_iter = 10
    elif max_iter <= 1000:
        show_iter = 50
    else:
        show_iter = 500

    if record_stats:
        all_time_losses_stats = analysis.getEmptyStatDic(max_iter)
        all_time_true_losses_stats = analysis.getEmptyStatDic(max_iter)
        
    if divergence == "reverse_kld_without_score_debug":
        all_stat_z = {}
        all_stat_z["median"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["high"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["higher"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))
        all_stat_z["max"] = numpy.zeros((max_iter, nfm.number_of_flows + 1))

    
    optimizer = torch.optim.Adam(nfm.parameters(), lr=learning_rate, weight_decay=0.0)

    nr_optimization_steps = 0

    current_best_true_loss = torch.inf

    for it in tqdm(range(max_iter)):

        optimizer.zero_grad()
        
        if annealing:
            beta_annealing = numpy.min([1., 0.01 + it / anneal_iter]) #  min(1, 0.01 + t/10000) is suggested in "Variational Inference with Normalizing Flows"
        else:
            beta_annealing = 1.0
        
        if divergence == "reverse_kld":
            loss, _, _ = core_adjusted.reverse_kld(nfm, num_mc_samples, beta=beta_annealing)
            true_loss = loss
        elif divergence == "reverse_kld_without_score":
            loss, true_loss, _, _, _ = core_adjusted.reverse_kld_without_score(nfm, num_mc_samples, beta=beta_annealing)
        elif divergence == "reverse_kld_without_score_debug":
            loss, true_loss,  z_stats_median, z_stats_high, z_stats_higher, z_stats_max = core_adjusted.reverse_kld_without_score_debug(nfm, num_mc_samples, beta=beta_annealing)
        else:
            assert(False)
        

        num_samples_after_filtering = loss.shape[0]
        assert(num_samples_after_filtering <= num_mc_samples)

        # records losses and statistics (median etc) of monte carlo samples from variational approximation
        if record_stats:
            analysis.logAllStats(all_time_losses_stats, it, loss.detach(), num_mc_samples - num_samples_after_filtering)
            commons.saveStatistics(all_time_losses_stats, "losses_stats")

            if divergence.startswith("reverse_kld_without_score"):
                analysis.logAllStats(all_time_true_losses_stats, it, true_loss.detach(),  num_mc_samples - num_samples_after_filtering)
                commons.saveStatistics(all_time_true_losses_stats, "true_losses_stats")

            if divergence == "reverse_kld_without_score_debug":
                if z_stats_median.shape[0] == 0:
                    all_stat_z["median"][it] = numpy.nan
                    all_stat_z["high"][it] = numpy.nan
                    all_stat_z["higher"][it] = numpy.nan
                    all_stat_z["max"][it] = numpy.nan
                else:
                    assert(z_stats_median.shape[0] == nfm.number_of_flows + 1)
                    all_stat_z["median"][it] = z_stats_median
                    all_stat_z["high"][it] = z_stats_high
                    all_stat_z["higher"][it] = z_stats_higher
                    all_stat_z["max"][it] = z_stats_max
        
        loss = torch.mean(loss)
        true_loss = torch.mean(true_loss)
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):

            if (it > (max_iter / 2)) and ~(torch.isnan(true_loss) | torch.isinf(true_loss)) and (true_loss < current_best_true_loss):
                # print("update best model")
                torch.save(nfm.state_dict(), commons.get_model_filename_best())
                current_best_true_loss = true_loss.detach().to('cpu')

            loss.backward()
            
            invalid_grad = False
            for name, param in nfm.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print("invalid gradients found ! SKIP")
                        invalid_grad = True
                        break
            
            if not invalid_grad:
                optimizer.step()
                nr_optimization_steps += 1

        if it % show_iter == 0:
            print(f"loss = {true_loss.to('cpu')} (without score = {loss.to('cpu')})")

        
    
    if record_stats:
        commons.saveStatistics(numpy.asarray([nr_optimization_steps]), "optSteps")

    if record_stats and divergence == "reverse_kld_without_score_debug":
        commons.saveStatistics(all_stat_z, "layer_z_stats")
        print("** SAVED layer_z_stats **")
    
    return nr_optimization_steps, current_best_true_loss
