
import torch

from normalizing_flows_core import FlowsMixture
import target_distributions

# this is an example


X, y, _ , _ = synthetic_data.lasso_linear(n = 20, d = 8)


setting["nr_mixture_components"] = 1
setting["init"] = "zeros"
setting["lr_exp"] = 4

setting["l2_strength"] = 0.0
setting["annealing"] = "no"

setting["data_type"] = "double"


setting["no_act_norm"] = "yes"
setting["nr_flows"] = nr_flows
setting["nr_cushions"] = 1
setting["loft_t"] = 100.0
setting["divergence"] = "reverse_kld_ws_debug"

# divergence = "reverse_kld_without_score", num_samples = 2 ** 8

setting["trainable_base"] = "no"
setting["cushion_type"] = "LOFT"
setting["realNVP_threshold"] = 0.1
setting["realNVP_variation"] = "var19"
setting["scaleShiftLayer"] = "ssL"
if method == "proposed_withStudentT":
    setting["use_student_base"] = "yes"

setting["max_iterations"] = 30000

args = getArgumentParser(**setting)

commons.DATA_TYPE = args.data_type
commons.setGPU()
torch.manual_seed(432432)

target, flows_mixture = initialize_target_and_flow(args, initialize)

VARIATIONAL_APPROXIMATION_TYPE = "RealNVP"

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

flows_mixture = FlowsMixture(target, nr_mixture_components, flow_type, number_of_flows, learn_mixture_weights, initial_loc_spec, use_student_base, use_LOFT, hidden_layer_size_spec)

model = target_distributions.LinearRegressionModel(3)

p = model.log_prior()
print("p = ", p)




# model1 = LinearRegressionModel(3)
# model2 = LinearRegressionModel(3)

# param_vec1 = torch.nn.utils.parameters_to_vector(model1.parameters())
# param_vec2 = torch.nn.utils.parameters_to_vector(model2.parameters())

# print("param_vec1 = ", param_vec1)
# print("param_vec2 = ", param_vec2)


# torch.nn.utils.vector_to_parameters(param_vec1, model2.parameters())

# param_vec1 = torch.nn.utils.parameters_to_vector(model1.parameters())
# param_vec2 = torch.nn.utils.parameters_to_vector(model2.parameters())

# print("param_vec1 = ", param_vec1)
# print("param_vec2 = ", param_vec2)

