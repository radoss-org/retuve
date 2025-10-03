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
Metric: Tonnis Classification of DDH
"""

import numpy as np

from retuve.draw import Overlay
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.hip_xray.utils import extend_line
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Colors
from retuve.utils import (
    find_perpendicular_point,
    point_above_below,
    point_left_right,
)


def _calculate_tonnis_grade(
    fem_point, pel_o, p_line_intercept, h_line_p1, h_line_p2, is_left_side
):
    """
    Determine the Tonnis grade for a single hip.

    Tonnis Classification:
    - Grade I: Femoral head medial to P-line & below SMA-line.
    - Grade II: Femoral head lateral to P-line & below SMA-line.
    - Grade III: Femoral head level with the SMA-line.
    - Grade IV: Femoral head is superior to the SMA-line.
    """
    # Define SMA line (parallel to H-line, passing through pel_o)
    h_line_vec = (h_line_p2[0] - h_line_p1[0], h_line_p2[1] - h_line_p1[1])
    sma_line_p2 = (pel_o[0] + h_line_vec[0], pel_o[1] + h_line_vec[1])

    # Check position relative to SMA line
    pos_vs_sma = point_above_below(pel_o, sma_line_p2, fem_point)

    if pos_vs_sma > 0:
        return 4  # Grade IV: Above SMA line
    elif pos_vs_sma == 0:
        return 3  # Grade III: On SMA line

    # If below SMA line, check position relative to Perkin's line
    # For left hip, medial is < 0. For right hip, medial is > 0.
    is_medial_to_p_line = (
        point_left_right(p_line_intercept, pel_o, fem_point) < 0
        if is_left_side
        else point_left_right(p_line_intercept, pel_o, fem_point) > 0
    )

    if is_medial_to_p_line:
        return 1  # Grade I: Medial to P-line
    else:
        return 2  # Grade II: Lateral to P-line


def find_tonnis(landmarks: LandmarksXRay) -> tuple[int, int]:
    """
    Calculate the Tonnis grade for both hips.

    :param landmarks: The Landmarks of the Hip
    :return: The Tonnis grade for both hips (1-4, 0 if invalid)
    """
    if not all(
        lm is not None
        for lm in [
            landmarks.pel_l_o,
            landmarks.pel_l_i,
            landmarks.pel_r_i,
            landmarks.pel_r_o,
            landmarks.fem_l,
            landmarks.fem_r,
        ]
    ):
        return 0, 0

    if np.any(
        (
            landmarks.fem_l == landmarks.pel_l_i,
            landmarks.fem_r == landmarks.pel_r_i,
        )
    ):
        return 0, 0

    h_line = (landmarks.pel_l_i, landmarks.pel_r_i)
    grade_left, grade_right = 0, 0

    if landmarks.fem_l is not None:
        p_line_int_left = find_perpendicular_point(*h_line, landmarks.pel_l_o)
        grade_left = _calculate_tonnis_grade(
            landmarks.fem_l,
            landmarks.pel_l_o,
            p_line_int_left,
            *h_line,
            is_left_side=True,
        )

    if landmarks.fem_r is not None:
        p_line_int_right = find_perpendicular_point(*h_line, landmarks.pel_r_o)
        grade_right = _calculate_tonnis_grade(
            landmarks.fem_r,
            landmarks.pel_r_o,
            p_line_int_right,
            *h_line,
            is_left_side=False,
        )

    return grade_left, grade_right


def _draw_tonnis_side(overlay, pel_o, h_line_p1, h_line_p2, grade_text):
    """Draws the SMA line and grade text for one side."""
    # Define SMA line (parallel to H-line, passing through pel_o) for drawing
    h_line_vec = (h_line_p2[0] - h_line_p1[0], h_line_p2[1] - h_line_p1[1])
    h_line_norm = np.linalg.norm(h_line_vec) * 0.015
    h_line_vec = (h_line_vec[0] / h_line_norm, h_line_vec[1] / h_line_norm)
    sma_line_p2 = (pel_o[0] + h_line_vec[0], pel_o[1] + h_line_vec[1])
    pel_o_original = pel_o
    pel_o = (pel_o[0] - h_line_vec[0], pel_o[1] - h_line_vec[1])

    sma_line_to_draw = extend_line(pel_o, sma_line_p2, scale=1.3, direction="both")

    overlay.draw_lines([sma_line_to_draw], color_override=Colors.LIGHT_GREEN)
    overlay.draw_text(grade_text, pel_o_original[0] - 100, pel_o_original[1] - 125)


def draw_tonnis(hip: HipDataXray, overlay: Overlay, config: Config):
    """
    Draw the Tonnis classification lines and grades on the X-Ray.

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

    if np.any(
        (
            landmarks.fem_l == landmarks.pel_l_i,
            landmarks.fem_r == landmarks.pel_r_i,
        )
    ):
        return overlay

    h_line = (landmarks.pel_l_i, landmarks.pel_r_i)

    if landmarks.fem_l is not None:
        _draw_tonnis_side(
            overlay,
            landmarks.pel_l_o,
            *h_line,
            f"Tonnis Grade: {hip.metrics[6].value}",
        )

    if landmarks.fem_r is not None:
        _draw_tonnis_side(
            overlay,
            landmarks.pel_r_o,
            *h_line,
            f"Tonnis Grade: {hip.metrics[7].value}",
        )

    return overlay
