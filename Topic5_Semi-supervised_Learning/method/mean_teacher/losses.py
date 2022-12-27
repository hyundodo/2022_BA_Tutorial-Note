"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


    
def JSD(input_logits, target_logits):
    input_softmax =  F.softmax(input_logits, dim=1)
    target_softmax=  F.softmax(target_logits, dim=1)

    m = 0.5 * (input_softmax + input_softmax)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(input_softmax, dim=1), m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(target_softmax, dim=1), m, reduction="batchmean") 
  
    return (0.5 * loss)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes
