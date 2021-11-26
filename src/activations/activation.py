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
    k = min([k, img.shape[-1], img.shape[-2]])
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
    def __init__(self, neighborhood_activation, is_batch_activation: bool, kernel_size: int):
        super(KernelActivation, self).__init__()
        self.neighborhood_activation = neighborhood_activation # applies activation to local image patch
        self.k = kernel_size
        self.is_batch_activation = is_batch_activation

    def forward(self, x):
        def apply(_patches, func):
            for patch in _patches:
                func(patch)
                yield patch

        # unfold into image patches -- should these overlap?
        patches, stitch_func = patcher(x, self.k)

        if self.is_batch_activation:
            patches = patches.squeeze()
            activations = self.neighborhood_activation(patches)
            activations = activations.unsqueeze(1)
        else:
            patches = patches.squeeze()
            pbar = tqdm(apply(patches, self.neighborhood_activation), total=patches.shape[0], desc="Activating", leave=False, position=1)
            activations = list(pbar)
            activations = torch.cat(activations, 0).unsqueeze(1)

        # fold back into single tensor
        repatched = stitch_func(activations)
        return repatched


def batch_nelu(patches: tensor, influence: float = 0.1) -> tensor:
    """
    Find the impact of the negative values in each patch, add a weighted sum to all the positive values and from
    those weighted values, activate the remaining positive values.

    Notes: Very Sparse, only positive values are passed through, should encourage all values in a patch to be positive
    if something exists in the patch.

    :param patches: Tensor of [N, H, W] where N is the number of patches, H is the patch height, and W is the patch width
    :param influence: The weight applied to the sum of negative values within the patch. 0 <= influence <= 1
    :return: Activated values in each patch, a tensor of size [N, H, W] with sparse activatons
    """

    # to find negative values fast, we want to set all positive values to 0, then sum the remaining items
    negative_values = patches.clone()
    negative_values[negative_values > 0] = 0

    # Take all the patches and sum them, then multiple their sum by the influence param
    impacts = negative_values.sum([1,2]) * influence

    # Make the list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    # Add the impact to all the patches.  The impact is guaranteed to be <= 0 by design, this means it will restrict
    # values more than regular ReLU.
    patches = patches + impacts

    # Set all negative values to 0, the non-linearity!
    patches[patches < 0] = 0

    return patches

def nelu(patch: tensor, influence: float = 0.1) -> tensor:
    # patch[patch > 0] += impact
    # patch[patch < 0] = 0

    impact = patch[patch < 0].sum() * influence
    torch.where(patch > 0, patch + impact, patch)

    z = torch.zeros_like(patch)
    torch.where(patch > 0, patch, z)
    return patch


def batch_passive_nelu(patches: tensor, influence: float = 0.1) -> tensor:
    """
    Find the impact of the negative values in each patch, add a weighted sum to all the positive values and from
    those weighted values, then allow all of the previously positive values through (even if they are negative now from
    the weighted sum)

    Notes: Less restrictive than NeLU, allows negative values to pass through, is as sparse as ReLU would be.

    :param patches: Tensor of [N, H, W] where N is the number of patches, H is the patch height, and W is the patch width
    :param influence: The weight applied to the sum of negative values within the patch. 0 <= influence <= 1
    :return: Activated values in each patch, a tensor of size [N, H, W] with sparse activations
    """

    # Get all the negative values by making all positive values equal to 0, then sum the patch
    negative_values = patches.clone()
    negative_values[negative_values > 0] = 0

    # Sum each patch and apply the influence weight to their sums
    impacts = negative_values.sum([1,2]) * influence

    # Make list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    # We want a index list of all the previously negative values (before we add the negative impacts to our positive
    # values).
    negative_indices = patches.clone()
    negative_indices[negative_indices <= 0] = 1

    # Add the impacts to all the values in each patch
    patches = patches + impacts

    # Zero out all the values that were not positive in the patch before adding the impact.
    patches[negative_indices == 1] = 0

    return patches


def passive_nelu(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch < 0] = 0
    patch[patch > 0] += impact
    return patch


def batch_accelerator(patches: tensor, influence: float = 0.1) -> tensor:
    """
    Find the impact of the positive values in the kernel, then remove some of the activation mass proportional from each
    positive activation and add that mass to the negative values TODO: each negative activation gets the full positive
    TODO: impact?? shouldn't this be proportional or even
    Anything left negative after adding the positive impacts is zeroed out.

    Notes: Less restrictive than ReLU

    :param patches: Tensor of [N, H, W] where N is the number of patches, H is the patch height, and W is the patch width
    :param influence: The weight applied to the sum of positive values within the patch. 0 <= influence <= 1
    :return: Activated values in each patch, a tensor of size [N, H, W] with sparse activations
    """

    # Get all the positive values by making all negative values equal to 0, then sum the patch
    positive_values = patches.clone()
    positive_values[positive_values < 0] = 0

    # Sum each patch and apply the influence weight to their sums
    impacts = positive_values.sum([1,2]) * influence

    # Make list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    # Remove the activation mass we are redistributing from each positive activation proportional to how much each
    # activation contributed
    positive_values *= 1 - influence

    # Get all the negative values by zeroing out the positive ones, then add the impacts
    negative_values = patches.clone()
    negative_values[negative_values > 0] = 0
    negative_values += impacts

    # add the positive values only tensor with the negative values only tensor (though the negative value tensor could
    # now contain positives because we added the impacts to them)
    patches = positive_values + negative_values

    # anything still below zero is now zeroed out.
    patches[patches < 0] = 0

    return patches


def accelerator(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch > 0].sum() * influence
    patch[patch > 0] *= 1 - influence
    patch[patch <= 0] += impact
    patch[patch < 0] = 0
    return patch


def batch_inhibitor(patches: tensor, influence: float = 0.1) -> tensor:
    """
    Find the impact of the negative values in the kernel, then remove some of the activation mass proportional from each
    negative activation and add that mass to the positive values TODO: each positive activation gets the full negative
    TODO: impact?? shouldn't this be proportional or even
    Anything left negative after adding the positive impacts is zeroed out.

    Notes: More restrictive than ReLU

    :param patches: Tensor of [N, H, W] where N is the number of patches, H is the patch height, and W is the patch width
    :param influence: The weight applied to the sum of negative values within the patch. 0 <= influence <= 1
    :return: Activated values in each patch, a tensor of size [N, H, W] with sparse activations
    """

    # Get all the negative values by making all negative values equal to 0, then sum the patch
    negative_values = patches.clone()
    negative_values[negative_values > 0] = 0

    # Sum each patch and apply the influence weight to their sums
    impacts = negative_values.sum([1,2]) * influence

    # Make list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    # Remove the activation mass we are redistributing from each negative activation proportional to how much each
    # activation contributed
    negative_values *= 1 - influence

    # Get all the positive values by zeroing out the negative ones, then add the impacts
    positive_values = patches.clone()
    positive_values[positive_values < 0] = 0
    positive_values += impacts

    # add the positive values only tensor with the negative values only tensor (though the positive value tensor could
    # now contain negatives because we added the impacts to them)
    patches = positive_values + negative_values

    # anything below zero is now zeroed out.
    patches[patches < 0] = 0

    return patches


def inhibitor(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch < 0] *= 1 - influence
    patch[patch >= 0] += impact
    patch[patch < 0] = 0
    return patch


def batch_excitator_v2(patches: tensor, influence: float = 0.1) -> tensor:
    # Get all the positive values by making all negative values equal to 0, then sum the patch
    positive_values = patches.clone()
    positive_values[positive_values < 0] = 0

    # Sum each patch and apply the influence weight to their sums
    impacts = positive_values.sum([1,2]) * influence

    # Make list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    non_positives = patches.clone()
    non_positives[patches > 0] = 0
    non_positives[patches <= 0] = 1
    non_positives = non_positives.sum([1,2]).unsqueeze(1).unsqueeze(1)
    non_positives[non_positives == 0] = 1

    # Remove the activation mass we are redistributing from each positive activation proportional to how much each
    # activation contributed
    positive_values *= 1 - influence

    # Get all the negative values by zeroing out the positive ones, then add the impacts
    negative_values = patches.clone()
    # Distribute the impacts based on the total number of negatives (equally distribute, this prevents the creation of "activation weight"
    negative_values += impacts / non_positives
    negative_values[patches > 0] = 0

    # add the positive values only tensor with the negative values only tensor (though the negative value tensor could
    # now contain positives because we added the impacts to them)
    patches = positive_values + negative_values

    # anything still below zero is now zeroed out.
    patches[patches < 0] = 0

    return patches

def batch_inhibitor_v2(patches: tensor, influence: float = 0.1) -> tensor:

    # Get all the negative values by making all negative values equal to 0, then sum the patch
    negative_values = patches.clone()
    negative_values[negative_values > 0] = 0

    # Sum each patch and apply the influence weight to their sums
    impacts = negative_values.sum([1,2]) * influence

    # Make list of impacts per patch match the dimensions of the patches ([N, 1, 1] where N is the number of patches)
    impacts = impacts.unsqueeze(1).unsqueeze(1)

    non_negatives = patches.clone()
    non_negatives[patches < 0] = 0
    non_negatives[patches >= 0] = 1
    non_negatives = non_negatives.sum([1,2]).unsqueeze(1).unsqueeze(1)
    # Prevent division by 0
    non_negatives[non_negatives == 0] = 1

    # Remove the activation mass we are redistributing from each negative activation proportional to how much each
    # activation contributed
    negative_values *= 1 - influence

    # Get all the positive values by zeroing out the negative ones, then add the impacts
    positive_values = patches.clone()
    positive_values[positive_values < 0] = 0
    positive_values += impacts

    # add the positive values only tensor with the negative values only tensor (though the positive value tensor could
    # now contain negatives because we added the impacts to them)
    patches = positive_values + negative_values

    # anything below zero is now zeroed out.
    patches[patches < 0] = 0

    return patches


def batch_softmax_relu(patches: tensor, threshold: float = 0.25) -> tensor:
    weights = torch.softmax(patches, dim=1)
    patches[weights < threshold] = 0
    return patches

def batch_max_relu(patches: tensor) -> tensor:
    maximums = patches.amax(dim=(1,2), keepdim=True)
    patches[patches < maximums] = 0
    patches[patches < 0] = 0
    return patches

def batch_max(patches: tensor) -> tensor:
    maximums = patches.amax(dim=(1,2), keepdim=True)
    patches[patches < maximums] = 0
    return patches

## TODO - make more at some point