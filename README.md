# LaProp-Optimizer
Codes accompanying the paper "LaProp: a Better Way to Combine Momentum with Adaptive Gradient"

The implementation is based on Pytorch. The optimizer ```laprop.LaProp``` uses the same calling signature compared with the standard ```optim.Adam``` in Pytorch, 
only with an additional optional argument ```centered = False``` controlling whether to use the centered second moment to divide the gradient.
