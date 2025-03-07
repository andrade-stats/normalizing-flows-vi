import torch

class IndexManager():

    # usage: name_of_parameter = number_of_dimensions
    def __init__(self, **param_to_nr_dim):
        print(param_to_nr_dim)
        self.param_to_nr_dim = param_to_nr_dim

        # get total number of dimensions
        self.d = 0
        for param in param_to_nr_dim:
            self.d += param_to_nr_dim[param]

        all_ids = torch.arange(self.d)
        self.param_to_ids = {}
        current_id = 0
        for param, dim in param_to_nr_dim.items():
            self.param_to_ids[param] = all_ids[current_id:(current_id+dim)]
            current_id += dim
        
        self.pos_constraint_ids = None


    def get_pos_constraint_ids(self, pos_param_list):

        pos_param_id_list = [self.param_to_ids[param] for param in pos_param_list]
        pos_constraint_ids = torch.cat(pos_param_id_list)
        return pos_constraint_ids
    
    def extract_samples(self, z, param):
        return z[:, self.param_to_ids[param]]
    
    def get_all_param_names(self):
        return self.param_to_ids.keys()