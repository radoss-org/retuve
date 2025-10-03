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
Metric: Alpha Angle

"""

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from radstract.math import smart_find_intersection

from retuve.classes.draw import Overlay
from retuve.classes.seg import SegObject
from retuve.hip_us.classes.general import HipDataUS, LandmarksUS
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import MetricUS
from retuve.utils import find_midline_extremes, warning_decorator

# To move the apex point left or right
APEX_RIGHT_FACTOR = 0


def find_alpha_landmarks(illium: SegObject, landmarks: LandmarksUS, config: Config):
    if illium is None or illium.midline_moved is None:
        return landmarks, 0

    # Get endpoints of main midline
    left_most, right_most = find_midline_extremes(illium.midline_moved)

    if left_most is None or right_most is None:
        return landmarks, 0

    # Convert to NumPy arrays once
    left_most = np.array(left_most, dtype=float)
    right_most = np.array(right_most, dtype=float)
    midline_moved = np.array(illium.midline_moved, dtype=float)

    # Precompute slope of main line
    dx = right_most[1] - left_most[1]
    dy = right_most[0] - left_most[0]
    if dx == 0:
        m = None  # vertical
    else:
        m = dy / dx

    # Orthogonal slope
    if m is None:
        m_orth = 0
    elif m == 0:
        m_orth = None
    else:
        m_orth = -1 / m

    # Sample every 5th point
    sampled_points = midline_moved[::5]

    results = []
    append_result = results.append  # local binding for speed

    # Precompute left/right tuple for intersection calls
    left_tuple = (left_most[1], left_most[0])
    right_tuple = (right_most[1], right_most[0])

    for py, px in sampled_points:
        a = px - 100
        if m_orth is None:
            # vertical orthogonal line
            p2 = (px, py + 100)
        else:
            p2 = (a, m_orth * (a - px) + py)

        try:
            closest_point = smart_find_intersection(
                left_tuple, right_tuple, (px, py), p2
            )

        except ValueError:
            closest_point = None

        if closest_point is not None:
            min_dist = np.hypot(closest_point[0] - px, closest_point[1] - py)
            append_result((min_dist, (py, px), (closest_point[1], closest_point[0])))

    if not results:
        return landmarks, 0

    # Convert to NumPy for fast argmax
    results_arr = np.array(results, dtype=object)
    # Find max distance, break ties by furthest right
    max_idx = max(range(len(results)), key=lambda i: (results[i][0], results[i][1][1]))

    max_distance, main_point, closest_point = results[max_idx]

    landmarks.apex = (main_point[1], main_point[0])
    landmarks.left = (left_most[1], left_most[0])
    landmarks.right = (right_most[1], right_most[0])

    angle = np.arctan2(
        landmarks.apex[1] - landmarks.left[1],
        landmarks.apex[0] - landmarks.left[0],
    )
    angle = np.degrees(angle)

    return landmarks, angle


@warning_decorator(alpha=True)
def find_alpha_angle(points: LandmarksUS) -> float:
    """
    Calculate the Alpha Angle.

    :param points: LandmarksUS: The landmarks object.

    :return: float: The Alpha Angle.
    """
    if not (points and points.left and points.apex and points.right):
        return 0

    # find angle ABC of points
    A, B, C = (
        np.array(points.left),
        np.array(points.apex),
        np.array(points.right),
    )
    AB = np.linalg.norm(A - B)
    BC = np.linalg.norm(B - C)
    AC = np.linalg.norm(A - C)
    angle = np.arccos((BC**2 + AB**2 - AC**2) / (2 * BC * AB))

    angle = np.degrees(angle)
    angle = round((180 - angle), 1)

    return round(angle, 2)


def draw_alpha(hip: HipDataUS, overlay: Overlay, config: Config) -> Overlay:
    """
    Draw the Alpha Angle on the Overlay.

    :param hip: HipDataUS: The Hip Data.
    :param overlay: Overlay: The Overlay.
    :param config: Config: The Config.

    :return: Overlay: The updated Overlay.
    """
    alpha = hip.get_metric(MetricUS.ALPHA)
    if alpha != 0:
        m1 = (hip.landmarks.apex[1] - hip.landmarks.left[1]) / (
            hip.landmarks.apex[0] - hip.landmarks.left[0]
        )
        m2 = (hip.landmarks.right[1] - hip.landmarks.apex[1]) / (
            hip.landmarks.right[0] - hip.landmarks.apex[0]
        )

        b1 = hip.landmarks.apex[1] - m1 * hip.landmarks.apex[0]
        b2 = hip.landmarks.apex[1] - m2 * hip.landmarks.apex[0]

        # x2 to account for difference in gradient
        new_right_apex = (
            hip.landmarks.apex[0] + 350,
            m1 * (hip.landmarks.apex[0] + 350) + b1,
        )
        new_right_landmark = (
            hip.landmarks.right[0],
            m2 * (hip.landmarks.right[0]) + b2,
        )

        overlay.draw_lines(
            (
                (hip.landmarks.left, new_right_apex),
                (hip.landmarks.apex, new_right_landmark),
            ),
        )

        if config.visuals.display_full_metric_names:
            title = "Alpha Angle"
        else:
            title = "a"

        overlay.draw_text(
            f"{title}: {alpha}",
            int(hip.landmarks.left[0]),
            int(hip.landmarks.left[1] - 40),
            header="h2",
        )

    return overlay


def bad_alpha(hip: HipDataUS) -> bool:
    """
    Check if the Alpha Angle is bad.

    :param hip: HipDataUS: The Hip Data.

    :return: bool: True if the Alpha Angle is bad.
    """

    if hip.get_metric(MetricUS.ALPHA) <= 0 or hip.get_metric(MetricUS.ALPHA) > 90:
        return True

    return False
