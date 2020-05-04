Here we present the files and commands we have used for the transformer tranining experiments. 
We use the implementation provided by the [Fairseq](https://github.com/pytorch/fairseq) package, and run the [training script](https://github.com/pytorch/fairseq/blob/master/fairseq_cli/train.py) thereof.

## Setup
To add LaProp to the Fairseq package to use, one needs to copy file ```laprop.py``` above into ```optim``` subdirectory of the installed Fairseq library. 
Then LaProp is invoked by the following commandline options using the training script above. 

```--optimizer laprop [--laprop-betas '(mu, nu)'] [--laprop-eps epsilon] [--wd weight_decay] [--laprop-centered]```

## Use
### IWSLT14 German to English Translation
Following the prescription given in https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md section **IWSLT'14 German to English (Transformer)**, we prepared the data in ```data-bin``` directory and ran the following command. The arguments concerning the choice between Adam and LaProp, or whether to use warmup, are indicated as exclusive choice ```[adam|laprop]``` and optional argument```[--warmup-updates num]```. We have plotted the training losses in our paper.
```
python3.6 train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    [--optimizer adam --adam-betas '(0.9, 0.98)' | --optimizer laprop  --laprop -betas '(0.9, 0.98)'] --clip-norm 0.0 \
    --lr 3e-4 --lr-scheduler polynomial_decay  [--warmup-updates 2000] --total-num-update 60000 \
    --dropout 0.3 --weight-decay 0.0001 --max-update 60000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096  --distributed-world-size 1 
```
