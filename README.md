# LaProp-Optimizer
Codes accompanying the paper "LaProp: a Better Way to Combine Momentum with Adaptive Gradient"

The implementation is based on Pytorch. The optimizer ```laprop.LaProp``` uses a similar calling signature compared with the standard ```optim.Adam``` in Pytorch, 
with an additional optional argument ```centered = False``` controlling whether to use a centered evaluation of the second moment at the adaptivity denominator.
