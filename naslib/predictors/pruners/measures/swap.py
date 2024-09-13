import numpy as np
import torch
import torch.nn as nn
from . import measure
from naslib.utils.utilities import count_parameters, network_weight_gaussian_init


def cal_regular_factor(model, mu, sigma):
    model_params = torch.as_tensor(count_parameters(model))
    regular_factor = torch.exp(-(torch.pow((model_params - mu), 2) / sigma))
    return regular_factor


class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        self.activations = self.activations.T  # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)

        del self.activations
        self.activations = None
        torch.cuda.empty_cache()

        return self.swap * regular_factor


class SWAP:
    def __init__(self, model=None, inputs=None, device='cuda', seed=9000, regular=True, mu=1.5, sigma=1.5):
        self.model = model
        self.interFeature = []
        self.seed = seed
        self.regular_factor = 1
        self.inputs = inputs
        self.device = device

        if regular and mu is not None and sigma is not None:
            self.regular_factor = cal_regular_factor(self.model, mu, sigma).item()

        self.reinit(self.model, self.seed)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.swap = SampleWiseActivationPatterns(self.device)

        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.swap = SampleWiseActivationPatterns(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0: return
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)
            self.swap.collect_activations(activtions)

            return self.swap.calSWAP(self.regular_factor)



@measure("swap", bn=True, copy_net=True, force_clean=True)
def compute_swap(net, inputs, outputs, device='cuda', seed=0, loss_fn=None, regular=False, mu=None, sigma=None, split_data=1):
    swap = SWAP(model=net, inputs=inputs, device=device, seed=seed, regular=False, mu=0, sigma=0)

    net = net.apply(network_weight_gaussian_init)
    swap.reinit()
    swap_score = swap.forward()
    swap.clear()

    return swap_score
#
# @measure("swap", bn=True, copy_net=True, force_clean=True)
# def compute_swap(net, inputs, outputs, device='cuda', seed=0, loss_fn=None, regular=False, mu=None, sigma=None, split_data=1):
#     # Initialize variables
#     interFeature = []
#     stored_activations = None
#     regular_factor = 1
#
#     # Regularization factor calculation
#     if regular and mu is not None and sigma is not None:
#         regular_factor = cal_regular_factor(net, mu, sigma).item()
#
#     # Set the seed for reproducibility
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
#
#     def reinit(model=None, new_seed=None, seed=None, interFeature=[]):
#         if model is not None:
#             model = model
#             register_hook(model, interFeature) #------------------------------------------- todo
#             swap = SampleWiseActivationPatterns(device)  #------------------------------------------- todo
#
#         if new_seed is not None and new_seed != seed:
#             seed = new_seed
#             torch.manual_seed(new_seed)
#             torch.cuda.manual_seed(new_seed)
#         del interFeature
#         interFeature = []
#         torch.cuda.empty_cache()
#         return swap
#
#
#     def clear(device, interFeature):
#         swap = SampleWiseActivationPatterns(device)
#         del interFeature
#         interFeature = []
#         torch.cuda.empty_cache()
#         return swap
#
#     def hook_in_forward(module, input, output, interFeature):
#         if isinstance(input, tuple) and len(input[0].size()) == 4:
#             interFeature.append(output.detach())
#             return interFeature
#
#     def register_hook(model, interFeature):
#         for n, m in model.named_modules():
#             if isinstance(m, nn.ReLU):
#                 m.register_forward_hook(hook=lambda mod, inp, out: hook_in_forward(mod, inp, out, interFeature))
#
#     swap = reinit(net, seed, seed, interFeature)
#     # Register hooks
#     register_hook(net, interFeature)
#
#     # Forward pass and collect activations
#     with torch.no_grad():
#         net.forward(inputs.to(device))
#
#     if len(interFeature) == 0:
#         return
#
#     activations = torch.cat([f.view(inputs.size(0), -1) for f in interFeature], 1)
#     stored_activations = swap.collect_activations(activations, stored_activations, device)
#
#     # Calculate SWAP
#     swap_value = swap.calSWAP(stored_activations, regular_factor)
#
#     return swap_value
