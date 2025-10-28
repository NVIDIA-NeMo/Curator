# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.spatial.transform import Rotation


def to_xcolor(color):
    """Convert RGB color to xcolor string."""
    r, g, b = [round(255 * x) for x in color]
    return f"rgb,255:red,{r};green,{g};blue,{b}"


def from_xcolor(color_str: str) -> tuple[float, float, float]:
    """Parse xcolor string back to RGB float values."""
    if color_str == "white":
        return 1.0, 1.0, 1.0
    elif color_str == "black":
        return 0.0, 0.0, 0.0
    elif not color_str.startswith("rgb,255:"):
        raise ValueError(f"Unsupported xcolor string: {color_str}")

    parts = color_str.replace("rgb,255:", "").split(";")
    rgb = {}
    for part in parts:
        color, value = part.strip().split(",")
        rgb[color] = int(value) / 255.0

    return rgb["red"], rgb["green"], rgb["blue"]


def hue_shift(color: list[float], shift: float) -> list[float]:
    """Shift the hue of an RGB color.
    :param color: the RGB color as a list of three scalars (0-1)
    :param shift: the hue shift in degrees (0-360)
    """
    # Create RGB rotation matrix
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(shift)
    rot = Rotation.from_rotvec(angle * axis).as_matrix()
    color = np.dot(rot, color).clip(0, 1)
    return color.tolist()


def contrast_ratio(color1: list[float], color2: list[float]) -> float:
    """Calculate the W3C WCAG contrast ratio.
    :param color1: the first RGB color as a list of three scalars (0-1)
    :param color2: the second RGB color as a list of three scalars (0-1)
    :return: the contrast ratio
    """
    # https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def relative_luminance(color: list[float]) -> float:
    """Calculate the relative luminance of an RGB color.
    :param color: the RGB color as a list of three scalars (0-1)
    :return: the relative luminance
    """
    # https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    r, g, b = color
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _srgb_to_linear(value: float) -> float:
    assert 0 <= value <= 1, f"Color value out of range: {value}"
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4
