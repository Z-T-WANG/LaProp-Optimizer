Here we present the files and commands that we have used for the transformer tranining experiments in the paper. 
We use the implementation provided by the [Fairseq](https://github.com/pytorch/fairseq) package, and use the training script thereof (https://github.com/pytorch/fairseq/blob/master/fairseq_cli/train.py).


###Use
To add our LaProp to the Fairseq package, one needs to copy the file ```laprop.py``` above into the ```optim``` folder in the installed Fairseq library. 
Then, LaProp can be invoked by the following commandline options when using the training script above. 

```--optimizer laprop [--laprop-betas '(mu, nu)'] [--laprop-eps epsilon] [--wd weight_decay] [--laprop-centered]```
