# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List

import nvidia.dali.fn as fn
from nvidia.dali.types import FLOAT, DALIInterpType
from timm.data.transforms import MaybeToTensor
from torchvision.transforms.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

ERROR_MESSAGE = """Transforms do not conform to expected style and cannot be automatically converted.
Expected:
    Compose(
        Resize(interpolation=bicubic or linear, max_size=None, antialias=True),
        CenterCrop(),
        MaybeToTensor(),
        Normalize(),
    )

Got: {}

Please manually convert the image transformations to use DALI
"""

# Linear = Bilinear and Cubic = Bicubic
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#nvidia.dali.types.DALIInterpType
SUPPORTED_INTERPOLATIONS = {
    InterpolationMode.BICUBIC: DALIInterpType.INTERP_CUBIC,
    InterpolationMode.BILINEAR: DALIInterpType.INTERP_LINEAR,
}


def convert_transforms_to_dali(torch_transform: Compose) -> List:
    """
    Converts a list of PyTorch/Timm image transformations into DALI transformations
    Only works with transformations that follow this pattern:

    Compose(
        Resize(interpolation=bicubic or bilinear, max_size=None, antialias=True),
        CenterCrop(),
        MaybeToTensor(),
        Normalize(),
    )

    Anything that does not follow this pattern will cause a ValueError to be raised
    """
    if not isinstance(torch_transform, Compose):
        raise ValueError(ERROR_MESSAGE.format(torch_transform))

    crop = None
    mean = [0.0]
    std = [1.0]
    resize_shorter = 0.0
    interp_type = DALIInterpType.INTERP_LINEAR

    # Loop over all transforms and extract relevant parameters
    for transform in torch_transform.transforms:
        if isinstance(transform, Resize):
            if transform.interpolation not in SUPPORTED_INTERPOLATIONS:
                raise ValueError(ERROR_MESSAGE.format(torch_transform))
            interp_type = SUPPORTED_INTERPOLATIONS[transform.interpolation]
            resize_shorter = transform.size
        elif isinstance(transform, CenterCrop):
            crop = transform.size
        elif isinstance(transform, Normalize):
            mean = transform.mean
            std = transform.std
        elif isinstance(transform, MaybeToTensor):
            continue
        else:
            raise ValueError(ERROR_MESSAGE.format(torch_transform))

    dali_transforms = [
        partial(
            fn.resize,
            device="gpu",
            interp_type=interp_type,
            resize_shorter=resize_shorter,
        ),
        # We need to multiply by 255 because DALI deals entirely in pixel values
        partial(
            fn.crop_mirror_normalize,
            device="gpu",
            crop=crop,
            dtype=FLOAT,
            mean=mean * 255,
            std=std * 255,
        ),
    ]
    return dali_transforms
