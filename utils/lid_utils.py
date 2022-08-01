from __future__ import absolute_import
from __future__ import print_function

import torch
import numpy as np
import collections
from tqdm import tqdm
from functools import partial

class LID():
    def __init__(self, model, normal, adv, noisy, batch_size, k):
        self.normal_X = normal
        self.adv_X = adv
        self.noisy_X = noisy
        self.model = model
        self.bs = batch_size
        self.k = k
        # get the number of layers
        self.lid_dim = 0
        # a dictionary that keeps saving the activations as they come
        self.activations = collections.defaultdict(list)
        # get deep representations
        for name, m in model.named_modules():
            # if type(m)==nn.Conv2d or type(m)==nn.Linear:
                # partial to assign the layer name to each hook
            m.register_forward_hook(partial(self.hook, name))
            self.lid_dim += 1

    def mle_batch(self, key, query, k):
        k = min(k, len(key)-1)
        f = lambda v: -k/np.sum(np.log((v+0.0000001)/(v[-1]+0.0000001)))
        dist = torch.cdist(query, key, 2) # (BS x BS)
        _sorted_dist, _ = torch.sort(dist, 1) # (BS x k)
        sorted_dist = _sorted_dist[:, 1:k+1].cpu()
        a = np.apply_along_axis(f, axis=1, arr=sorted_dist) # (BS, )
        return a

    def hook(self, name, _, __, output):
        self.activations[name].extend(output.detach().cpu())

    def get_features(self, model, data):
        _ = model(data)
        features = [torch.cat(features, 0) for features in self.activations.values()]
        return features

    def estimate(self, step):
        start = self.bs*step
        end = np.minimum(self.bs*(step+1), len(self.normal_X))
        n_feed = end - start
        # create empty array for saving lid
        lid_batch = np.zeros(shape=(n_feed, self.lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, self.lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, self.lid_dim))
        # extract activation map from total layers
        normal_features = self.get_features(self.model, self.normal_X[start:end])
        # print("INPUT:", self.normal_X[start:end])
        self.activations = collections.defaultdict(list)
        adv_features = self.get_features(self.model, self.adv_X[start:end])
        self.activations = collections.defaultdict(list)
        noisy_features = self.get_features(self.model, self.noisy_X[start:end])
        self.activations = collections.defaultdict(list)
        for i, (normal, adv, noisy) in enumerate(zip(normal_features, adv_features, noisy_features)):
            # feature flatten
            normal, adv, noisy = (features.to(self.adv_X.device).view(n_feed, -1) for features in (normal, adv, noisy))
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = self.mle_batch(normal, normal, k=self.k)
            # print("lid_batch: ", lid_batch.shape)
            lid_batch_adv[:, i] = self.mle_batch(normal, adv, k=self.k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            lid_batch_noisy[:, i] = self.mle_batch(normal, noisy, k=self.k)
            # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return lid_batch, lid_batch_noisy, lid_batch_adv

    def get_lids_random_batch(self):
        """
        Get the local intrinsic dimensionality of each Xi in X_adv
        estimated by k close neighbours in the random batch it lies in.
        :param model:
        :param X: normal images
        :param X_noisy: noisy images
        :param X_adv: advserial images    
        :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures  
        :param k: the number of nearest neighbours for LID estimation  
        :param batch_size: default 100
        :return: lids: LID of normal images of shape (num_examples, lid_dim)
                lids_adv: LID of advs images of shape (num_examples, lid_dim)
        """
        print("Number of layers to estimate: ", self.lid_dim)
        lids = []
        lids_adv = []
        lids_noisy = []

        n_batches = int(np.ceil(self.normal_X.size(0) / float(self.bs)))
        for i_batch in tqdm(range(n_batches)):
            lid_batch, lid_batch_noisy, lid_batch_adv = self.estimate(i_batch)
            lids.extend(lid_batch)
            lids_adv.extend(lid_batch_adv)
            lids_noisy.extend(lid_batch_noisy)

        lids = np.asarray(lids, dtype=np.float32)
        lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
        lids_adv = np.asarray(lids_adv, dtype=np.float32)

        return lids, lids_noisy, lids_adv
