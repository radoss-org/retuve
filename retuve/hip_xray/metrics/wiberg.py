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
Metric: Wiberg Index
"""

from retuve.draw import Overlay
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.hip_xray.utils import extend_line
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Colors
from retuve.utils import angle_between_lines, find_perpendicular_point


def find_wiberg(landmarks: LandmarksXRay) -> tuple[float, float]:
    """
    Calculate the Wiberg Index for both hips.

    :param landmarks: The Landmarks of the Hip
    :return: The Wiberg Index for both hips
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

    if landmarks.h_point_l is None:
        wiberg_index_left = 0
    else:
        P_line_intercept_left = find_perpendicular_point(
            landmarks.pel_l_i, landmarks.pel_r_i, landmarks.pel_l_o
        )

        P_line_left = (
            landmarks.pel_l_o,
            P_line_intercept_left,
        )

        wiberg_line_left = (
            landmarks.pel_l_o,
            landmarks.h_point_l,
        )
        wiberg_index_left = angle_between_lines(P_line_left, wiberg_line_left)

    if landmarks.h_point_r is None:
        wiberg_index_right = 0
    else:
        P_line_intercept_right = find_perpendicular_point(
            landmarks.pel_r_i, landmarks.pel_l_i, landmarks.pel_r_o
        )
        P_line_right = (
            landmarks.pel_r_o,
            P_line_intercept_right,
        )

        wiberg_line_right = (
            landmarks.pel_r_o,
            landmarks.h_point_r,
        )

        wiberg_index_right = angle_between_lines(P_line_right, wiberg_line_right)

    wiberg_index_left = round(wiberg_index_left, 1)
    wiberg_index_right = round(wiberg_index_right, 1)

    return wiberg_index_left, wiberg_index_right


def draw_wiberg(hip: HipDataXray, overlay: Overlay, config: Config):
    """
    Draw the Wiberg Index on the X-Ray.

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

    if landmarks.h_point_l is not None:
        P_line_intercept_left = find_perpendicular_point(
            landmarks.pel_l_i, landmarks.pel_r_i, landmarks.pel_l_o
        )
        new_P_line_left = extend_line(
            landmarks.pel_l_o,
            P_line_intercept_left,
            scale=3,
            direction="both",
        )
        new_wiberg_line_left = extend_line(
            landmarks.pel_l_o, landmarks.h_point_l, scale=2, direction="down"
        )

        overlay.draw_lines(
            [
                new_P_line_left,
            ],
            color_override=Colors.RED,
        )
        overlay.draw_lines(
            [
                new_wiberg_line_left,
            ],
            color_override=Colors.GOLD_LIGHT,
        )

        overlay.draw_text(
            f"Wiberg: {hip.metrics[2].value}",
            landmarks.pel_l_o[0] - 100,
            (landmarks.pel_l_i[1] + 80),
        )

    if landmarks.h_point_r is not None:
        P_line_intercept_right = find_perpendicular_point(
            landmarks.pel_r_i, landmarks.pel_l_i, landmarks.pel_r_o
        )
        new_P_line_right = extend_line(
            P_line_intercept_right,
            landmarks.pel_r_o,
            scale=3,
            direction="both",
        )
        new_wiberg_line_right = extend_line(
            landmarks.pel_r_o, landmarks.h_point_r, scale=2, direction="down"
        )

        overlay.draw_lines(
            [
                new_P_line_right,
            ],
            color_override=Colors.RED,
        )
        overlay.draw_lines(
            [
                new_wiberg_line_right,
            ],
            color_override=Colors.GOLD_LIGHT,
        )
        overlay.draw_text(
            f"Wiberg: {hip.metrics[3].value}",
            landmarks.pel_r_o[0] - 100,
            (landmarks.pel_r_i[1] + 80),
        )

    return overlay
