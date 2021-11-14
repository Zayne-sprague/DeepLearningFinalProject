import torch
from torch import nn
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F
from torch import tensor
from typing import Union, Callable
from functools import partial
from tqdm import tqdm


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
        def apply(_patches, func):
            for patch in _patches:
                func(patch)
                yield patch

        # unfold into image patches -- should these overlap?
        patches, stitch_func = patcher(x, self.k)

        ## TODO - make this faster some how
        pbar = tqdm(apply(patches, self.neighborhood_activation), total=patches.shape[0], desc="Activating", leave=False, position=1)
        activations = list(pbar)
        activations = torch.cat(activations, 0).unsqueeze(1)

        # fold back into single tensor
        repatched = stitch_func(activations)
        return repatched


## TODO - make these faster (parallel maybe?)

def nelu(patch: tensor, influence: float = 0.1) -> tensor:
    # patch[patch > 0] += impact
    # patch[patch < 0] = 0

    impact = patch[patch < 0].sum() * influence
    torch.where(patch > 0, patch + impact, patch)
    torch.where(patch > 0, 1, 0)
    return patch

## TODO - make these work like nelu

def passive_nelu(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch < 0] = 0
    patch[patch > 0] += impact
    return patch


def accelerator(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch > 0].sum() * influence
    patch[patch > 0] *= 1 - influence
    patch[patch <= 0] += impact
    patch[patch < 0] = 0
    return patch


def inhibitor(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch < 0] *= 1 - influence
    patch[patch >= 0] += impact
    patch[patch < 0] = 0
    return patch

## TODO - make more at some point