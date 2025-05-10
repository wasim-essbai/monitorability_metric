import torch
from scipy import integrate
import numpy as np

def extract_activations(model, dataloader, layer_index, device):
    model.eval()
    all_activations = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out, features = model(x, return_feature_list=True)
            pred_values = torch.max(out, 1).indices
            features = features[-1]
            features = features[pred_values == y, :]
            y = y[pred_values == y]

            all_activations += features.detach().cpu().numpy().tolist()
            all_labels += y.detach().cpu().numpy().tolist()

    all_activations = np.array(all_activations)     
    all_labels = np.array(all_labels) 
    return all_activations, all_labels

def compute_mds(feats, mean, precision):
    feats = np.atleast_2d(feats)
    distances = np.einsum('ij,jk,ik->i', feats - mean, precision, feats - mean)    
    return distances if feats.shape[0] > 1 else distances[0]

def uniform_dist(x, ua, la):
    result = []
    for i in x:
        if la <= i <= ua:
            result.append(1 / (ua - la))
        else:
            result.append(0)
    return np.array(result)

def get_integral(y, x):
    ua = np.max(x)
    la = np.min(x)

    y_tol = y
    y_int = y_tol * uniform_dist(x, ua, la)
    return integrate.trapezoid(y_int, x)

def fgsm_attack(image, epsilon, data_grad, clamp=True):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad

    if clamp:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def perform_fgsm_attack(model, data, target, epsilon, loss_function, clamp=True, denorm=None, normalize_function=None):
    data.requires_grad = True

    pred = model(data) #pred does not have requires_grad=True
    loss = loss_function(pred, target)

    model.zero_grad()

    loss.backward()

    data_grad = data.grad.data
    data_denorm = data
    if denorm is not None:
        data_denorm = denorm(data)

    perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad, clamp)
    if normalize_function is not None:
        perturbed_data = normalize_function(perturbed_data)
    return perturbed_data