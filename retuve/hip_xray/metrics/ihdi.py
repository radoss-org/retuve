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
Metric: IHDI Classification of DDH
"""

from radstract.math import smart_find_intersection

from retuve.draw import Overlay
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.hip_xray.utils import extend_line
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Colors
from retuve.utils import (
    find_perpendicular_point,
    point_above_below,
    point_left_right,
    rotate_in_place,
)


def _get_ihdi_lines(pel_o, pel_i, h_line_p1, h_line_p2, rotation_angle):
    """Calculate the geometric lines required for IHDI classification."""
    p_line_intercept = find_perpendicular_point(h_line_p1, h_line_p2, pel_o)
    ihdi_intercept = smart_find_intersection(
        h_line_p1, h_line_p2, pel_o, p_line_intercept
    )
    d_line = rotate_in_place(ihdi_intercept, pel_i, angle_deg=rotation_angle)
    return p_line_intercept, d_line


def _calculate_grade(
    h_point,
    pel_o,
    p_line_intercept,
    d_line,
    h_line_p1,
    h_line_p2,
    is_left_side,
):
    """
    Determine the IHDI grade for a single hip based on its landmarks.

    IHDI Classification:
    - Grade I: H-point is at or medial to the P-line
    - Grade II: H-point is lateral to the P-line and at or medial to the D-line
    - Grade III: H-point is lateral to the D-line and at or inferior to the H-line
    - Grade IV: H-point is superior to the H-line
    """
    # Grade IV: H-point is superior to the H-line (Hilgenreiner's line)
    if point_above_below(h_line_p1, h_line_p2, h_point) > 0:
        return 4

    # Check position relative to P-line (Perkin's line)
    # For the left hip, medial is < 0. For the right hip, medial is > 0.
    is_medial_to_p_line = (
        point_left_right(p_line_intercept, pel_o, h_point) < 0
        if is_left_side
        else point_left_right(p_line_intercept, pel_o, h_point) > 0
    )

    # Grade I: H-point is at or medial to the P-line
    if is_medial_to_p_line:
        return 1

    is_medial_to_d_line = point_above_below(d_line[0], d_line[1], h_point) < 0

    # Grade II: Medial to D-line
    if is_medial_to_d_line:
        return 2
    # Grade III: Lateral to D-line
    else:
        return 3


def find_ihdi(landmarks: LandmarksXRay) -> tuple[int, int]:
    """
    Calculate the IHDI grade for both hips.

    :param landmarks: The Landmarks of the Hip
    :return: The IHDI grade for both hips (1-4, 0 if invalid)
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

    h_line = (landmarks.pel_l_i, landmarks.pel_r_i)
    grade_left, grade_right = 0, 0

    if landmarks.h_point_l is not None:
        p_line_int_l, d_line_l = _get_ihdi_lines(
            landmarks.pel_l_o, landmarks.pel_l_i, *h_line, 135
        )
        grade_left = _calculate_grade(
            landmarks.h_point_l,
            landmarks.pel_l_o,
            p_line_int_l,
            d_line_l,
            *h_line,
            is_left_side=True,
        )

    if landmarks.h_point_r is not None:
        p_line_int_r, d_line_r = _get_ihdi_lines(
            landmarks.pel_r_o, landmarks.pel_r_i, *h_line, 225
        )
        grade_right = _calculate_grade(
            landmarks.h_point_r,
            landmarks.pel_r_o,
            p_line_int_r,
            d_line_r,
            *h_line,
            is_left_side=False,
        )

    return grade_left, grade_right


def _draw_side(overlay: Overlay, d_line, grade_text, text_pos):
    """Draws the D-line and grade text for one side."""
    new_d_line = extend_line(d_line[0], d_line[1], scale=2.5, direction="down")
    overlay.draw_lines([new_d_line], color_override=Colors.WHITE)
    overlay.draw_text(grade_text, text_pos[0] - 100, text_pos[1] - 100)


def draw_ihdi(hip: HipDataXray, overlay: Overlay, config: Config):
    """
    Draw the IHDI classification lines and grades on the X-Ray.

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

    h_line = (landmarks.pel_l_i, landmarks.pel_r_i)

    if landmarks.h_point_l is not None:
        _, d_line_left = _get_ihdi_lines(
            landmarks.pel_l_o, landmarks.pel_l_i, *h_line, 135
        )
        _draw_side(
            overlay,
            d_line_left,
            f"IHDI Grade: {hip.metrics[4].value}",
            landmarks.pel_l_o,
        )

    if landmarks.h_point_r is not None:
        _, d_line_right = _get_ihdi_lines(
            landmarks.pel_r_o, landmarks.pel_r_i, *h_line, 225
        )
        _draw_side(
            overlay,
            d_line_right,
            f"IHDI Grade: {hip.metrics[5].value}",
            landmarks.pel_r_o,
        )

    return overlay
