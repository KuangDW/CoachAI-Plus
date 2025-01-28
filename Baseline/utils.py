import torch
import torch.nn as nn

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)

def get_prob_from_q(q_values, mask=None):

    prob = q_values - q_values.min() + 1e-5
    prob = prob / prob.sum()
    prob = prob.cpu()

    if mask is None:
        return prob
    
    masked_prob = prob.masked_fill(~mask, 0.0)
    sum_prob = masked_prob.sum(dim=-1, keepdim=True)
    prob = torch.where(sum_prob > 0, masked_prob / sum_prob, mask.float() / mask.sum())
    return prob
