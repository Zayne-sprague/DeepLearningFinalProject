import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from typing import Tuple


class Inhibitor(nn.Module):
    def __init__(self, kernel_size=3, activated=1, inhibitance=0.2):
        super(Inhibitor, self).__init__()
        self.k = _pair(kernel_size)
        self.activated = activated
        self.inhibitance = inhibitance

    def forward(self, x):
        act = torch.where(x > self.activated, 1, 0)
        return x * act
        #
        # if len(x.shape) == 4:
        #     return self.conv_forward(x)
        # else:
        #     return self.dense_forward(x)

    def dense_forward(self, x):
        batch_size = x.shape[0]
        neurons = x.shape[1]

        for b in range(batch_size):
            for n in range(neurons):

                act = x[b, n]

                # if x[b,n] < act:
                #     x[b,n] = 0

                if act > self.activated:
                    # x[b,n] = 0

                    lhs = max(0, n - self.k[1])
                    rhs = min(neurons, n + self.k[1])

                    x[b, lhs:rhs] = 0
                    x[b, n] = act
                    n = min(rhs, neurons-1)

                    # for kn in range(lhs, rhs):
                    #     act += self.inhibitance * x[b, kn]
                    #     x[b, kn] *= 1 - self.inhibitance
        return x


    def conv_forward(self, x):
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]


        for b in range(0, batch_size):
            for c in range(0, channel_size):
                for h in range(0, height):
                    for w in range(0, width):
                        act = x[b,c,h,w]

                        # if act < self.activated:
                        #     x[b,c,h,w] = 0

                        if act > self.activated:
                            # x[b,c,h,w] = 0

                            lhs = max(0, w - self.k[1])
                            rhs = min(width, w + self.k[1])

                            ths = max(0, h - self.k[0])
                            bhs = min(height, h+self.k[0])

                            # for kh in range(ths, bhs):
                            #     for kw in range(lhs, rhs):
                            #         act += self.inhibitance * x[b,c,kh,kw]
                            #         x[b,c,kh,kw] *= 1 - self.inhibitance

                            h = min(ths, height-1)
                            w = min(rhs, width-1)
                            x[b,c,ths:bhs,lhs:rhs] = 0
                            x[b,c,h,w] = act


        return x
