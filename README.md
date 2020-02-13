# LaProp-Optimizer
Codes accompanying the paper [LaProp: a Better Way to Combine Momentum with Adaptive Gradient](https://arxiv.org/abs/2002.04839)

## Use
This implementation is based on Pytorch. The LaProp optimizer is the class ```LaProp``` in file ```laprop.py```. ```laprop.LaProp``` uses the same calling signature as the standard ```optim.Adam``` of Pytorch, 
only with an additional optional argument ```centered = False``` controlling whether to compute the centered second moment to divide the gradient.

The learning rate and the weight decay are decoupled in ```laprop.LaProp```, and therefore when one wants to apply a learning rate schedule, one needs to decay both ```'lr'``` and ```'weight_decay'``` stored in the optimizer. 

When ```centered``` is enabled, the optimizer will update for ```self.steps_before_using_centered = 10``` steps in the non-centered way to accumulate information of the gradient, and after that it starts to use the centered strategy. The number of the non-centered steps is tentatively set to 10 at its initialization.

## Additional Details compared with the Paper
In ```laprop.LaProp```, we have combined the learning rate and the accumulated momentum into one term, so that when the learning rate changes, the momentum accumulated by a larger learning rate still has a larger effect. 

The bias correction terms are treated similarly. Especially, the momentum bias correction is computed from the learning rate and the momentum hyperparameter at each step, so that the bias correction is guaranteed in the presence of a changing learning rate and momentum parameter; the squared gradient bias correction is only computed from the beta2 hyperparameter at each step and does not involve the learning rate.
