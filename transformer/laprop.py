# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import types

import torch
import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('laprop')
class FairseqLaprop(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)
        if torch.cuda.is_available():
            self._optimizer = LaProp(params, **self.optimizer_config)
        else:
            self._optimizer = LaProp(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--laprop-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for LaProp optimizer')
        parser.add_argument('--laprop-eps', type=float, default=1e-15, metavar='D',
                            help='epsilon for LaProp optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--laprop-centered', action='store_true', 
                            help='centered second moment')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.laprop_betas),
            'eps': self.args.laprop_eps,
            'weight_decay': self.args.weight_decay,
            'centered': self.args.laprop_centered,
        }


class LaProp(torch.optim.Optimizer):
    """Implements LaProp algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, centered = False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, centered=centered)
        super(LaProp, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('LaProp does not support sparse gradients')
                amsgrad, centered = group['amsgrad'], group['centered']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if centered:
                        state['exp_mean_avg_beta2'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if centered:
                        state['exp_mean_avg_beta2'] = state['exp_mean_avg_beta2'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if centered:
                    exp_mean_avg_beta2 = state['exp_mean_avg_beta2']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq
                if centered:
                    exp_mean_avg_beta2.mul_(beta2).add_(1 - beta2, grad)
                    if state['step']>10:
                        squared_mean = exp_mean_avg_beta2 ** 2
                        denom = denom - squared_mean

                if amsgrad:
                    if state['step']>10  or not centered: 
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = denom.sqrt().add_(group['eps'])

                state['exp_avg_lr'] = state['exp_avg_lr'] * beta1 + (1 - beta1) * group['lr']

                bias_correction1 = state['exp_avg_lr'] / group['lr'] if group['lr']!=0. else 1. #1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = 1 / bias_correction1

                step_of_this_grad = grad / denom * math.sqrt(bias_correction2)
                exp_avg.mul_(beta1).add_( (1 - beta1) * group['lr'], step_of_this_grad)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
