'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
from . import measure
from ..p_utils import label_sim_matrix

def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # print(mod.weight.grad.data.size())
                # print(mod.weight.data.size())
                if mod.weight.grad is not None:
                    grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


@measure("zicoLGA", bn=True, copy_net=True, force_clean=True)
def compute_zico(net, inputs, outputs, device='cuda', seed=0, loss_fn=None, split_data=1):
    sim_matrix0 = label_sim_matrix(inputs, outputs[0])
    sim_matrix0 = torch.tensor(sim_matrix0).to(device)
    sim_matrix1 = label_sim_matrix(inputs, outputs[1])
    sim_matrix1 = torch.tensor(sim_matrix1).to(device)
    grad_dict = {}
    # net.zero_grad() todo should be added?
    net.train()

    net.to(device)
    for i, _ in enumerate(inputs): #trainloader is actually composed of only two batches
        net.zero_grad()
        data, label = inputs[i], outputs[i]
        data, label = data.to(device), label.to(device)

        logits = net(data)
        loss = loss_fn(logits, label)
        loss.backward()
        grad_dict = getgrad(net, grad_dict, i)

    res = caculate_zico(grad_dict)
    mean_label_sim0 = torch.mean(sim_matrix0)
    res = res / torch.norm(sim_matrix0-mean_label_sim0)
    mean_label_sim1 = torch.mean(sim_matrix1)
    res = res / torch.norm(sim_matrix1-mean_label_sim1)
    return res.cpu()