# Momentum Capsule Network

Official implementation of the paper [Momentum Capsule Networks (MoCapsNet)](). 

## Abstract

```
Capsule networks are a class of neural networks that achieved
promising results on many computer vision tasks. However, baseline cap-
sule networks have failed to reach state-of-the-art results on more com-
plex datasets due to the high computation and memory requirements.
We tackle this problem by proposing a new network architecture, called
Momentum Capsule Network (MoCapsNet). MoCapsNets are inspired
by Momentum ResNets, a type of network that applies reversible resid-
ual building blocks. Reversible networks allow for recalculating activa-
tions of the forward pass in the backpropagation algorithm, so those
memory requirements can be drastically reduced. In this paper, we pro-
vide a framework on how invertible residual building blocks can be ap-
plied to capsule networks. We will show that MoCapsNet beats the ac-
curacy of baseline capsule networks on MNIST, SVHN and CIFAR-10
while using considerably less memory. 
```

## Acknowledgments

The CapsNet implementation is based on the implementation of [Daniel Havir's](https://github.com/danielhavir/capsule-network) repository.

