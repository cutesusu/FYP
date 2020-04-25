import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for name, module in model.named_children():
        if name in ['fc1', 'fc2']:     
            print("==== layer in weight sharing now: %s=====" %(name))
            print(module)
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            print("Begin Kmeans!")
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            #print("kmeans over!")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        else:
            if name == 'stn':
                for subname, submodule in module.named_modules():
                    # do not weight share fc_loc.2, it causes min arg empty error
                    if subname in ['fc_loc.0']:
                        print("==== layer in weight sharing now: %s=====" %(subname))
                        print(submodule)
                        dev = submodule.weight.device
                        weight = submodule.weight.data.cpu().numpy()
                        shape = weight.shape
                        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
                        min_ = min(mat.data)
                        max_ = max(mat.data)
                        space = np.linspace(min_, max_, num=2**bits)
                        print("Begin Kmeans!")
                        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
                        #print("kmeans over!")
                        kmeans.fit(mat.data.reshape(-1,1))
                        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                        mat.data = new_weight
                        submodule.weight.data = torch.from_numpy(mat.toarray()).to(dev)

