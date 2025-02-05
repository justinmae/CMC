import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        if torch.cuda.is_available():
            self.multinomial.cuda()
        self.K = K

        # Using register_buffer so that it is not trained by optimizer
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item() # 0.5
        batchSize = l.size(0) # 256
        outputSize = self.memory_l.size(0) # 5000
        inputSize = self.memory_l.size(1) # 64

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1) # 256*(4096+1) first col is positive
            idx.select(1, 0).copy_(y.data) # set first col to be positive
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        out_ab = torch.exp(torch.div(out_ab, T))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l = torch.exp(torch.div(out_l, T))

        # set Z_0 if haven't been set yet,
        # Z_0 is used as a constant approximation of Z, to scale the probs
        if Z_l < 0:
            self.params[2] = out_l.mean() * outputSize
            Z_l = self.params[2].clone().detach().item()
            print("normalization constant Z_l is set to {:.1f}".format(Z_l))
        if Z_ab < 0:
            self.params[3] = out_ab.mean() * outputSize
            Z_ab = self.params[3].clone().detach().item()
            print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

        # compute out_l, out_ab
        out_l = torch.div(out_l, Z_l).contiguous()
        out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab
