import torch
import numpy as np
import scipy
import scipy.stats


def bundle_entropy(a, y, num_classes):
    """ Calculate the bundle entropy (normalized by bundle size)
        for a given layer output a with corresponding label y.
        Note: The function torch.isclose is used to evaluate whether 
        two samples are equal. This is a slightly different approximation 
        than the methodology as proposed by https://arxiv.org/abs/2011.02956.

        param a: Output batch of a given layer
        param y: Corresponding label as int

        returns: num_bundles, bundle_entropy
    """
    a = a.flatten(start_dim=1)
    dim = a.shape[1]
    batch_size = a.shape[0]
    bundle = [batch_size-i for i in range(batch_size)]

    #
    # Calculate bundles
    #
    for i in range(len(a)-1):
        already_bundleed = bundle[i] != batch_size-i
        if already_bundleed:
            continue

        for j in range(i+1, len(a)):
            equal_components = torch.isclose(a[i], a[j]).int().sum()
            if equal_components != dim:
                continue

            bundle[j] = bundle[i]

    unique_bundle_ids = np.unique(bundle)
    num_bundles = len(unique_bundle_ids)

    #
    # Calculate normalized bundle entropy
    #
    bundle_entropy_norm = 0.0
    for i in unique_bundle_ids:
        pos = [p for p, c in enumerate(bundle) if c == i]
        bundle_size = float(len(pos))

        label_occurence = torch.FloatTensor([y[c_pos] for c_pos in pos])
        label_occurence = _get_occurences(label_occurence, num_classes)

        bundle_entropy = scipy.stats.entropy(label_occurence)
        bundle_entropy_norm += bundle_entropy * bundle_size / batch_size

    return num_bundles, bundle_entropy_norm


def _get_occurences(tensor, num_classes):
    bundle_size = float(tensor.size()[0])
    ret = [0.0] * num_classes

    for i in range(num_classes):
        ret[i] = np.sum([1.0 for c in tensor if c == i]) / bundle_size

    return ret
