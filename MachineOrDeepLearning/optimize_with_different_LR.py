"""
To show optimize different trainable parameters with different learning rates.

Generally, an optimize operation contains 2 steps, 
computer gradients and apply gradients, for both torch and tensorflow. 
Here is an torch example of apply different learning rates to different layers.

This trick is useful when learning rates matter much, 
for example, BERT + CRF model for token classification,
if the same learing rate is used, CRF layer will not be fully trained, 
so a much larger LR is needed, e.g. 1000 times of BERT LR.

Futher, if some paramters need to be fixed, just remove it from the var list.
"""

import torch 

class Optimizer:
    def __init__(self, learning_rate, name="SGD", **kwargs):
        self._name = name
        self._iterations = 0
        self._learning_rate = learning_rate
        self._clipper = None 
    
    def compute_gradients(self, loss, var_list, aggregate=False):
        """
        return gradients
        """
        var_list = list(var_list)
        grads = [v.grad if v is not None else None for v in var_list]

        self.detach_gradients(grads)

        if not aggregate:
            self.zero_gradients(grads)

        loss.backward()
        return [v.grad if v is not None else None for v in var_list]
    
    def apply_gradients(self, grads_and_vars):
        """
        grads_and_vars = zip(gradients, list(model.named_parameters()))
        """
        self._iterations += 1
        lr = self._learning_rate
        grads, var_list = list(zip(*grads_and_vars))

        if self._clipper is not None:
            pass  # in case grad is too large or too small.

        for grad, var in zip(grads, var_list):
            if grad is None:
                continue

            # Convert if grad is not FP32
            grad = grad.data.float()
            name, var = var  # unpack var to get name

            # attention that lr is decreasing with iterations.
            # need add a step to process lr
            step_size = lr

            """
            change the step_size if necessary
            """
            if "crf" in name:  
                step_size *= 1000
            
            # apply gradient
            if var.dtype == torch.float32:
                var.data.add_(grad, alpha=-step_size)
            else:
                fp32_var = var.data.float()
                fp32_var.add_(grad, alpha=-step_size)
                var.data.copy_(fp32_var)
