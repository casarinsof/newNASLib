import torch
from sklearn.decomposition import PCA
import copy
import math

from . import measure
from ..p_utils import get_layer_metric_array, label_sim_matrix, gram_matrix



@measure('lga', bn=True)
def compute_lga(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    sim_matrix = label_sim_matrix(inputs, targets)
    sim_matrix = torch.tensor(sim_matrix).to(device)

    net.zero_grad()
    output = net(inputs)
    last_grad = []
    all_grad = []

    for _idx in range(len(inputs)):
        output[_idx:_idx+1].backward(torch.ones_like(output[_idx:_idx+1]), retain_graph=True) # todo ???? output.backward?
        grad = []

        for name, W in net.named_parameters():
            if 'weight' in name and W.grad is not None and 'classifier' not in name:
                grad.append(W.grad.view(-1).detach())

        all_grad.append(copy.deepcopy(torch.cat(grad, -1)))

        net.zero_grad()
        torch.cuda.empty_cache()

    all_grad = torch.stack(all_grad, 0)
    ntk = torch.einsum('nc, mc->nm', [all_grad, all_grad])
   # knas = torch.mean(ntk)

   # eig_val, _ = torch.eig(ntk, eigenvectors=False)
   # eig_val = eig_val.reshape(-1)
   # eig_val = eig_val[eig_val.nonzero()]
   # if eig_val.size()[0] == 0:
   #     tenas = torch.tensor(1e-4)
   # else:
   #     tenas = torch.max(eig_val) / torch.min(eig_val)

   # f_norm = torch.norm(ntk)

    mean_ntk = torch.mean(ntk)
    mean_label_sim = torch.mean(sim_matrix)

    lgc_numerator = torch.einsum('ij, ij->ij', [ntk-mean_ntk, sim_matrix-mean_label_sim]).sum() #.sum() #((ntk-mean_ntk) * (sim_matrix-mean_label_sim))
    lgc_denominator = (torch.norm(ntk-mean_ntk) * torch.norm(sim_matrix-mean_label_sim))
    lgc_score = lgc_numerator / lgc_denominator

    if math.isnan(lgc_score):
        lgc_score = torch.tensor(1e-4)

    net.zero_grad()

    return lgc_score.cpu() # knas, tenas, f_norm