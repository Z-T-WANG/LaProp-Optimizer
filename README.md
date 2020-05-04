# LaProp-Optimizer
Codes accompanying the paper [LaProp: a Better Way to Combine Momentum with Adaptive Gradient](https://arxiv.org/abs/2002.04839)

## Use
This implementation is based on Pytorch. The LaProp optimizer is the class ```LaProp``` in file ```laprop.py```, which is adapted from the standard optimizer ```optim.Adam``` of Pytorch. ```laprop.LaProp``` uses the same calling signature as the standard ```optim.Adam```,
only with an additional optional argument ```centered = False``` controlling whether to compute the centered second moment instead of the squared gradient. 
The input argument ```betas``` corresponds to the tuple <img src="https://github.com/Z-T-WANG/LaProp-Optimizer/blob/master/images/cde5b07ebaa7f798b2ed9abf1799672d.png" /> in our paper.

The learning rate and the weight decay are decoupled in ```laprop.LaProp```, and therefore when one wants to apply a learning rate schedule with weight decay, one needs to decay ```'lr'``` and ```'weight_decay'``` simultaneously in the optimizer. 

When ```centered``` is enabled, the optimizer will update for ```self.steps_before_using_centered = 10``` steps in the non-centered way to accumulate information of the gradient, and after that it starts to use the centered strategy. The number of the non-centered steps is tentatively set to 10 at its initialization.

## Additional Details compared with the Paper
In ```laprop.LaProp```, we have combined the learning rate and the accumulated momentum into one term, so that when the learning rate changes, the momentum accumulated by a larger learning rate still has a larger effect. 

The bias correction terms are treated similarly. Especially, the momentum bias correction is computed from the learning rate and the momentum hyperparameter at each step, so that the bias correction is guaranteed in the presence of a changing learning rate and momentum parameter; the squared gradient bias correction is only computed from the beta2 hyperparameter at each step and does not involve the learning rate.

## Future Work
We plan to put our codes for training of MNIST, CIFAR10, IMDB into this repository which are described in our paper, and we will also include the codes that are adapted for Fairseq training of IWSLT and Roberta.

## Citation
If you use LaProp in your research, please cite our paper with the following bibtex item
```
@article{ziyin2020laprop,
  title={LaProp: a Better Way to Combine Momentum with Adaptive Gradient},
  author={Ziyin, Liu and Wang, Zhikang T and Ueda, Masahito},
  journal={arXiv preprint arXiv:2002.04839},
  year={2020}
}
```
