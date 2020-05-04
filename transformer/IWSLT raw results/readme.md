The raw results of the transformer based IWSLT translation are presented. The laprop and adam training both use ```beta_2=0.98```
and ```init_lr=3e-4```. If a warmup schedule is used, a "warmup" keyword is added to the file name.

The tuples ```(raw training loss, perplexity, norm of the gradient)``` during training are recorded in the files. Note that our recorded raw 
training loss is higher than the standard NLL loss because label smoothing is applied during the training.
