We use the codes provided in https://github.com/bearpaw/pytorch-classification to implement the CIFAR10 classification training, 
and we change the optimizer in the script to the ```AdamW``` or ```LaProp``` class in the above ```optimizers.py``` file. The command we run is basically
```
python3.6 cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --lr 1e-3
```
with the momentum, the beta2 term, and the optimizer specified as in the paper.
