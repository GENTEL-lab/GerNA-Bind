import warnings
import functools
from collections import defaultdict

class CustomOptimizer:

    def __init__(self,params,defaults):

        self.defaults = defaults
        self.state = defaultdict(dict)

        self.param_groups=[]
        param_groups = list(params)

        if len(param_groups)==0:
            raise ValueError('optimizer got an empty parameter list')

        if not isinstance(param_groups[0], dict):
            param_groups = [{'params':param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        """
        作用是将param_group放进self.param_groups中
        param_group是字典，Key是params，Value是param_groups=list(params)
        """

        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']

        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)
    
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
    
        params = param_group['params']

        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                    "in future, this will cause an error; ",stacklevel=3)
    
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
  
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def step(self, closure):
        raise NotImplementedError

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string
    
import math

class CustomAdam(CustomOptimizer):
    def __init__(self, params,lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False,maximize=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(CustomAdam, self).__init__(params, defaults)
  
    def __setstate__(self, state):
        super(CustomAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
  
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                if group['maximize']:
                    grad = -p.grad.data
                else:
                    grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                        'please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']        

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

            if amsgrad:
              # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
              # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss