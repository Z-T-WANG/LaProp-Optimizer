Here we present the files and commands we have used for the transformer tranining experiments. 
We use the implementation provided by the [Fairseq](https://github.com/pytorch/fairseq) package, and run the [training script](https://github.com/pytorch/fairseq/blob/master/fairseq_cli/train.py) thereof.

## Setup
To add LaProp to the Fairseq package to use, one needs to copy file ```laprop.py``` above into ```optim``` subdirectory of the installed Fairseq library. 
Then LaProp is invoked by the following commandline options using the training script above. 

```--optimizer laprop [--laprop-betas '(mu, nu)'] [--laprop-eps epsilon] [--wd weight_decay] [--laprop-centered]```

## Use
### IWSLT14 German to English Translation
Following the prescription given in https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md **IWSLT'14 German to English (Transformer)** section, we prepared the data in ```data-bin``` directory and ran the following command. The arguments concerning the choice between Adam and LaProp, or whether to use warmup, are indicated as exclusive choice ```[adam|laprop]``` and optional argument```[--warmup-updates num]```. We have plotted the training losses in our paper.
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

### RoBERTa Pretraining
Referring to the prescription given in https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md , we prepared the full English Wikipedia as the training data, which is detailed in the Appendix of our paper, and we used the following command.
```
TOTAL_UPDATES=20000     # Our number of training steps
EXPECTED_UPDATES=125000 # The original number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0001          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=6         # Number of sequences per batch (batch size)
UPDATE_FREQ=21          # Increase the batch size 21x
DATA_DIR=data-bin/wikitext

python3.6 train.py --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    [--optimizer adam --adam-betas '(0.9,[0.98 | 0.999])' --adam-eps 1e-6 | --optimizer laprop --laprop-betas '(0.9,[0.98 | 0.999])'] \
    [--warmup-updates $WARMUP_UPDATES] --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $EXPECTED_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --distributed-world-size 1 \
    --save-interval 10  --seed 1 --no-epoch-checkpoints --no-last-checkpoints  --no-save \
    --disable-validation --mask-whole-words
```
