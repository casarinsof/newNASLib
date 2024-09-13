import numpy as np
import torch
import torch.nn as nn

def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

# def build_subnetwork_from_genotype(args, n_classes, supernet, genotype):
#     subnetwork = Network(args.init_channels, n_classes, args.layers, None, genotype) #todo modificare
#     # Copy weights from the supernet to the subnetwork
#     subnetwork_state_dict = subnetwork.state_dict()
#     supernet_state_dict = supernet.state_dict()
#
#     for key in subnetwork_state_dict.keys():
#         if key in supernet_state_dict.keys():
#             subnetwork_state_dict[key] = supernet_state_dict[key]
#
#     subnetwork.load_state_dict(subnetwork_state_dict)
#     return subnetwork
#
# def darts_random_sample(args, n=1000):
#     criterion = nn.CrossEntropyLoss()
#     if args.dataset == 'cifar100':
#         n_classes = 100
#     else:
#         n_classes = 10
#     model = DartsNetwork(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
#
#     genotypes = []
#     for i in range(n):
#         genotype = model.genotype()
#         genotypes.append(genotype)
#     return genotypes


if __name__ == '__main__':
    pass


