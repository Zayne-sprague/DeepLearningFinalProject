import torch
from torch import tensor
from typing import Union, Callable
from functools import partial

def nelu(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    # patch[patch > 0] += impact
    patch[patch > 0] = patch[patch > 0] + impact

    patch[patch < 0] = 0
    return patch


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

def excitator2(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch > 0].sum() * influence

    num_non_positives = len(patch <= 0)
    if num_non_positives == 0:
        return patch

    patch[patch > 0] *= 1 - influence
    patch[patch <= 0] += impact / num_non_positives

    patch[patch < 0] = 0
    return patch


def tmp1(patch: tensor) -> tensor:
    # Max relu in patch
    patch[patch < 0] = 0
    patch[patch < torch.max(patch)] = 0
    return patch


def inhibitor(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence
    patch[patch < 0] *= 1 - influence
    patch[patch >= 0] += impact
    patch[patch < 0] = 0
    return patch

def inhibitor2(patch: tensor, influence: float = 0.1) -> tensor:
    impact = patch[patch < 0].sum() * influence

    num_of_positives = len(patch >= 0)
    if num_of_positives == 0:
        patch[patch < 0] = 0
        return patch

    patch[patch < 0] *= 1 - influence
    patch[patch > 0] += impact / num_of_positives
    patch[patch < 0] = 0
    return patch

def sftmx_tmp(patch: tensor, threshold: float = 0.5) -> tensor:
    weights = torch.softmax(patch, dim=0)
    patch[weights < threshold] = 0
    return patch


def patcher(img: tensor, k: int = 2) -> Union[tensor, Callable]:
    batch_size = img.shape[0]
    patches: tensor = img.unfold(1, 1, 1).unfold(2, k, k).unfold(3, k, k)
    unfold_shape = patches.shape
    patches = patches.contiguous().view(-1, 1, k, k)

    return patches, partial(stitcher, batch_size=batch_size, unfold_shape=unfold_shape)

def stitcher(patches: tensor, batch_size, unfold_shape) -> tensor:
    # repatched = patches.contiguous().view(-1, 1, k, k).reshape(-1, c, h, w)
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(batch_size, output_c, output_h, output_w)

    return patches_orig


if __name__ == "__main__":
    from src.dataloader import get_dataloder
    from src.paths import DATA_DIR
    from PIL import Image
    import numpy as np
    from torchvision.transforms.functional import to_pil_image

    RUN_TEST_ACTIVATIONS = True
    DO_IMAGES = False

    if DO_IMAGES:

        train_dataloader, test_dataloader = get_dataloder("CIFAR-100", 3, DATA_DIR)

        for batch_index, (images, labels) in enumerate(train_dataloader):
            for i in images:
                img = to_pil_image(i)
                img = img.resize([256, 256])
                img.show()

            k = 8

            c = images[0].shape[0]
            h = images[0].shape[1]
            w = images[0].shape[2]

            patches, stitch_func = patcher(images, k)

            # MUTATE_PATCHES

            stitched = stitch_func(patches)

            for i in stitched:
                simg = to_pil_image(i)
                simg = simg.resize([256,256])
                simg.show()
            # for idx, patch in enumerate(patcher(images[0], 8)):
            #     pimg = to_pil_image(patch)
            #     pimg = pimg.resize([64, 64])
            #     pimg.show()
            #
            #     if idx > 10:
            #         break
            tmp = 1
            break


    if RUN_TEST_ACTIVATIONS:
        patch_size = [3,3]
        patch: tensor = torch.rand(patch_size) - 0.5

        print("ORIGINAL:")
        print(patch)
        print("\n\n")

        nelu_patch = nelu(patch.clone(), influence=0.1)

        print("NELU:")
        print(nelu_patch)
        print("\n\n")

        passive_nelu_patch = passive_nelu(patch.clone(), influence=0.1)

        print("PASSIVE NELU:")
        print(passive_nelu_patch)
        print("\n\n")

        inhibitor_patch = inhibitor(patch.clone(), influence=0.33)

        print("INHIBITOR:")
        print(inhibitor_patch)
        print("\n\n")

        inhibitorv2_patch = inhibitor2(patch.clone(), influence=0.33)

        print("INHIBITOR v2:")
        print(inhibitorv2_patch)
        print("\n\n")


        tmp1_patch = tmp1(patch.clone())

        print("TMP1:")
        print(tmp1_patch)
        print("\n\n")

        softmax_tmp_patch = sftmx_tmp(patch.clone(), threshold=0.33)

        print("SOFTMAX TMP:")
        print(softmax_tmp_patch)
        print("\n\n")


        accelerator_patch = accelerator(patch.clone(), influence=0.2)

        print("EXCITATOR v1:")
        print(accelerator_patch)
        print("\n\n")

        excitator2_patch = excitator2(patch.clone(), influence=0.2)

        print("Excitator v2:")
        print(excitator2_patch)
        print("\n\n")