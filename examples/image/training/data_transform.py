# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage, Resize


def get_train_transform(image_size: int = None):
    """
    Get training transforms.
    
    Args:
        image_size: Target image size. If None, no resizing is done (for CIFAR-10).
                   For ImageNet with DiT, typically use 64 or 32.
    """
    transform_list = [
        ToImage(),
    ]
    if image_size is not None:
        transform_list.append(Resize((image_size, image_size), antialias=True))
    transform_list.extend([
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ])
    return Compose(transform_list)
