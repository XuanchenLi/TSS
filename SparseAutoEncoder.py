import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, input):
        encoded = F.sigmod(self.encoder(input))
        decoded = F.sigmod(self.decoder(encoded))
        return encoded, decoded
    RHO = 0.01


def kl_divergence(p, q):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2