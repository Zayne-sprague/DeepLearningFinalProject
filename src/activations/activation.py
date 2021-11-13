import torch
from torch import nn
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F
from torch import tensor
from typing import Union, Callable
from functools import partial


def patcher(img: tensor, k: int = 2) -> Union[tensor, Callable]:
    batch_size = img.shape[0]
    patches: tensor = img.unfold(1, 1, 1).unfold(2, k, k).unfold(3, k, k)
    unfold_shape = patches.shape
    patches = patches.contiguous().view(-1, 1, k, k)

    return patches, partial(stitcher, batch_size=batch_size, unfold_shape=unfold_shape)


def stitcher(patches: tensor, batch_size, unfold_shape) -> tensor:
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(batch_size, output_c, output_h, output_w)

    return patches_orig


class KernelActivation(nn.Module):
    def __init__(self, neighborhood_activation, kernel_size):
        super(KernelActivation, self).__init__()
        self.neighborhood_activation = neighborhood_activation # applies activation to local image patch
        self.k = kernel_size

    def forward(self, x):
        if len(x.shape) == 4:
            # image tensor
            b, c, h, w = x.shape

            # unfold into image patches -- should these overlap?
            # patches = x.unfold(2, self.k, self.k).unfold(3, self.k, self.k).transpose(1, 3).reshape(-1, c, h, w)
            # patches = x.unfold(1, self.k, self.k).unfold(2, self.k, self.k).reshape(-1, self.k, self.k)
            patches, stitch_func = patcher(x, self.k)

            # apply kernel activation to each patch
            # activations = torch.tensor([self.neighborhood_activation(patch) for patch in patches])
            activations = torch.cat([self.neighborhood_activation(patch) for patch in patches], 0).unsqueeze(1)

            # fold back into single tensor
            # return F.fold(activations, x.shape[-2:], kernel_size=self.k)
            repatched = stitch_func(activations)
            return repatched
        else:
            # how would the dense activation work? do we look at a sliding window over a single vector of activations?
            return x


class InhibitorLocal(nn.Module):
    def __init__(self, activated=1, inhibitance=0.2):
        super(InhibitorLocal, self).__init__()
        self.activated = activated
        self.inhibitance = inhibitance

    def forward(self, x):
        return x
        # act = torch.where(x > self.activated, 1, 0)
        # return x * act

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
        height = x.shape[2] # this will be kernel-sized
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

def nelu(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch > 0] += impact
    patch[patch < 0] = 0
    return patch
