import numpy as np
import sklearn.covariance

from utils.mira_utils import *

class MIRA(object):
    def __init__(self, model, 
                 layer_index, 
                 epsilon_min, 
                 epsilon_max, 
                 n_values, 
                 loss_function, 
                 device,
                 data_norm=None, 
                 data_denorm=None, 
                 clamp=True):
        self.model = model
        self.layer_index = layer_index
        self.eplison_min = epsilon_min
        self.eplison_max = epsilon_max
        self.n_values = n_values
        self.loss_function = loss_function
        self.data_norm = data_norm
        self.data_denorm = data_denorm
        self.clamp = clamp
        self.device = device
        self.class_means = {}
        self.class_precisions = {}

    def fit(self, data_loader):
        all_activations, all_labels = extract_activations(self.model, data_loader, self.layer_index, self.device)
        for i in set(all_labels):
            class_activations = all_activations[all_labels == i]
            self.class_means[i] = class_activations.mean(axis=0)

            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(class_activations - class_activations.mean(axis=0))

            self.class_precisions[i] = group_lasso.precision_
    
    def get_evaluation_values(self, n_values):
        step = (self.eplison_max - self.eplison_min) / n_values
        values = np.arange(self.eplison_min, self.eplison_max, step)
        return np.append(values, self.eplison_max)
    
    def compute_all_distances(self, data_loader):
        all_distances = []
        evaluation_range = self.get_evaluation_values(self.n_values)
        for epsilon in evaluation_range:
            with torch.no_grad():
            
                epsilon_distances = []
                for data, target in data_loader:
                    (x, y) = (data.to(self.device), target.to(self.device))

                    x_tilde = perform_fgsm_attack(self.model, x, y, epsilon, self.loss_function, 
                                        self.clamp, self.data_denorm, self.data_norm)
                    out, feat_list = self.model(x_tilde, return_feature_list=True)
                    feats = feat_list[self.layer_index]
                    for i in set(target):
                        class_feats = feats[y==i]
                        class_distances = compute_mds(class_feats.detach().cpu().numpy(), self.class_means[i], self.class_precisions[i])
                        epsilon_distances += class_distances.tolist()
                epsilon_distances = np.array(epsilon_distances)
                all_distances.append(epsilon_distances.mean())
        return all_distances, evaluation_range
    
    def compute_mira_score(self, data_loader):
        all_distances, evaluation_range = self.compute_all_distances(data_loader)
        mira_score = get_integral(all_distances, evaluation_range)
        
        return mira_score

