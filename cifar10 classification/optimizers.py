from torch.optim import Optimizer
import math
import torch

class AdamW(torch.optim.Adam):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    if not 'max_exp_avg_sq' in state: 
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss

class LaProp(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15,
                 weight_decay=0, amsgrad=False, centered=False):
        self.centered = centered
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(LaProp, self).__init__(params, defaults)

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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = state['exp_avg_lr_2'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_mean_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if self.centered:
                    state['exp_mean_avg_sq'] = state['exp_mean_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1 - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1 - beta2)

                bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr']!=0. else 1. #1 - beta1 ** state['step']
                step_size = 1 / bias_correction1

                bias_correction2 = state['exp_avg_lr_2']
                
                denom = exp_avg_sq
                if self.centered:
                    exp_mean_avg_sq.mul_(beta2).add_(1 - beta2, grad)
                    if state['step']>5:
                        mean = exp_mean_avg_beta2 ** 2
                        denom = denom - mean

                if amsgrad:
                    if not (self.centered and state['step']<=5): 
                        # Maintains the maximum of all (centered) 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq

                denom = denom.div(bias_correction2).sqrt_().add_(group['eps'])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_( (1 - beta1) * group['lr'], step_of_this_grad)
                
                p.data.add_(-step_size, exp_avg )
                if group['weight_decay'] != 0:
                    p.data.add_( - group['weight_decay'], p.data)

        return loss


class LaProp_stable(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15,
                 weight_decay=0, amsgrad=False, centered=False):
        #betas = (betas, 1 - (1 - betas) ** 2)
        self.centered = centered
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(LaProp3, self).__init__(params, defaults)

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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # used only if centered
                    state['exp_mean_avg_beta2'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_mean_avg_beta2 = state['exp_avg'], state['exp_avg_sq'], state['exp_mean_avg_beta2']

                if amsgrad:
                    if not 'max_exp_avg_sq' in state: 
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = (1 - beta1 ** state['step'])
                bias_correction2 = (1 - (beta2 ** (state['step'] - 1)))

                denom = exp_avg_sq
                if self.centered and state['step']>5:
                    mean = exp_mean_avg_beta2 ** 2
                    denom = denom - mean

                if amsgrad:
                    if not (self.centered and state['step']<=5): 
                        # Maintains the maximum of all (centered) 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq

                c = (1.-beta1) ** 2
                bias_correction3 = c + (1. - c) * bias_correction2

                squared_grad = grad ** 2 
                denom = (c*squared_grad).add_(1.-c, denom).div_(bias_correction3).sqrt_().add_(group['eps'])
                step_of_this_grad = grad / denom
                
                exp_avg.mul_(beta1).add_(1. - beta1, step_of_this_grad)
                
                step_size = group['lr'] / bias_correction1

                p.data.add_(-step_size, exp_avg)
                if group['weight_decay'] != 0:
                    p.data.add_( - group['weight_decay'], p.data)

                # update the exponential average of squared gradients
                exp_avg_sq.mul_(beta2).add_((1. - beta2), squared_grad)
                if self.centered:
                    exp_mean_avg_beta2.mul_(beta2).add_(1. - beta2, grad)

        return 0, 0, 0, 0
