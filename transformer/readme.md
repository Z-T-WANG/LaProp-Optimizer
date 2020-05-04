Here we present the files and commands we have used for the transformer tranining experiments. 
We use the implementation provided by the [Fairseq](https://github.com/pytorch/fairseq) package, and run the [training script](https://github.com/pytorch/fairseq/blob/master/fairseq_cli/train.py) thereof.

## Use
To add LaProp to the Fairseq package to use, one needs to copy the file ```laprop.py``` above into the ```optim``` subdirectory of the installed Fairseq library. 
Then LaProp can be invoked by the following commandline options when using the training script above. 

```--optimizer laprop [--laprop-betas '(mu, nu)'] [--laprop-eps epsilon] [--wd weight_decay] [--laprop-centered]```
