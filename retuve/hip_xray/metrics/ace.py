# Copyright 2024 Adam McArthur
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

"""
Metric: Acentabular Index (ACE Index)
"""

from retuve.draw import Overlay
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.hip_xray.utils import extend_line
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Colors
from retuve.utils import angle_between_lines


def find_ace(landmarks: LandmarksXRay) -> tuple[float, float]:
    """
    Calculate the Acentabular Index (ACE Index) for both hips.

    :param landmarks: The Landmarks of the Hip
    :return: The ACE Index for both hips
    """
    if not all(
        lm is not None
        for lm in [
            landmarks.pel_l_o,
            landmarks.pel_l_i,
            landmarks.pel_r_i,
            landmarks.pel_r_o,
        ]
    ):
        return 0, 0

    H_line = (
        landmarks.pel_l_i,
        landmarks.pel_r_i,
    )

    ace_line_left = (
        landmarks.pel_l_o,
        landmarks.pel_l_i,
    )
    ace_line_right = (
        landmarks.pel_r_i,
        landmarks.pel_r_o,
    )

    ace_index_left = angle_between_lines(H_line, ace_line_left)

    ace_index_right = angle_between_lines(H_line, ace_line_right)

    ace_index_left = round(ace_index_left, 1)
    ace_index_right = round(ace_index_right, 1)

    return ace_index_left, ace_index_right


def draw_ace(hip: HipDataXray, overlay: Overlay, config: Config):
    """
    Draw the Acentabular Index (ACE Index) on the X-Ray.

    :param hip: The Hip Data
    :param overlay: The Overlay
    :param config: The Config

    :return: The Drawn Overlay
    """
    landmarks = hip.landmarks

    if not all(
        lm is not None
        for lm in [
            landmarks.pel_l_o,
            landmarks.pel_l_i,
            landmarks.pel_r_i,
            landmarks.pel_r_o,
        ]
    ):
        return overlay

    new_H_line = extend_line(landmarks.pel_l_i, landmarks.pel_r_i, scale=1.7)
    new_A_line = extend_line(
        landmarks.pel_l_o, landmarks.pel_l_i, scale=1.7, direction="up"
    )
    new_B_line = extend_line(
        landmarks.pel_r_o, landmarks.pel_r_i, scale=1.7, direction="up"
    )

    overlay.draw_lines([new_H_line], color_override=Colors.PURPLE)
    overlay.draw_lines([new_A_line, new_B_line])

    overlay.draw_text(
        f"ACE Index: {hip.metrics[0].value}",
        landmarks.pel_l_o[0] - 100,
        (landmarks.pel_l_o[1] - 75),
    )

    overlay.draw_text(
        f"ACE Index: {hip.metrics[1].value}",
        landmarks.pel_r_o[0] - 100,
        (landmarks.pel_r_o[1] - 75),
    )

    return overlay
