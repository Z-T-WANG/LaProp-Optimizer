# Rainbow

An implementation of Rainbow in PyTorch, debugged and modified from https://github.com/belepi93/pytorch-rainbow 

## Use
To reproduce our results, please run the following command
```
python main.py --env BreakoutNoFrameskip-v4 --multi-step 1 --double --noisy \
--prioritized-replay --frame-stack 4 --lr 0.0000625 --c51 --train-freq 4 \
--max-frames 12000000 --buffer-size 500000 --optim laprop --update-target 10000 \
--learning-start 80000 --sigma-init 0.5 --alpha 0.5 --eps_final 0.005
```
To use Adam as the optimizer, one can change ```--optim laprop``` to ```--optim adam```. The trained models are saved in directory ```models``` in the format of ```{optimizer}_{epoch}.pth```. To evaluate the models, by adding a command ```--evaluate```, the script will read all models under directory ```models``` and only evaluate the saved models that start with {optimizer} as provided in ```--load-model {optimizer}```. For example, to evaluate the models trained by Adam, one should use ```--evaluate --load-model adam```.

## Acknowledgements
- Kalxhin(https://github.com/Kaixhin/NoisyNet-A3C)
- higgsfield(https://github.com/higgsfield)
- openai(https://github.com/openai/baselines)
