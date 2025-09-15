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
Metric: Coverage
"""

import numpy as np
from networkx import diameter
from radstract.math import smart_find_intersection

from retuve.classes.draw import Overlay
from retuve.classes.seg import SegObject
from retuve.hip_us.classes.general import HipDataUS, LandmarksUS
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import MetricUS
from retuve.utils import warning_decorator

FEM_HEAD_SCALE_FACTOR = 0.18


def find_cov_landmarks(
    femoral: SegObject, landmarks: LandmarksUS, config: Config
) -> LandmarksUS:
    """
    Algorithm to find the landmarks for the Coverage metric.

    :param femoral: SegObject: The femoral SegObject.
    :param landmarks: LandmarksUS: The landmarks object.
    :param config: Config: The Config

    :return: LandmarksUS: The updated landmarks object.
    """
    if not (
        femoral
        and any(m in config.hip.measurements for m in MetricUS.ALL())
        and femoral.points is not None
        and landmarks.apex is not None
    ):
        return landmarks

    points = femoral.points

    # Find the most right and left points
    most_right_point = max(points, key=lambda x: x[0])
    most_left_point = min(points, key=lambda x: x[0])

    top_most_point = min(points, key=lambda x: x[1])
    bottom_most_point = max(points, key=lambda x: x[1])

    # find diameter using right-left and top-bottom
    diameter_1 = abs(most_right_point[0] - most_left_point[0])
    diameter_2 = abs(bottom_most_point[1] - top_most_point[1])

    if diameter_1 == 0 or diameter_2 == 0:
        return landmarks

    # Reject frames with a non-circular femoral head
    if abs((diameter_1 - diameter_2) / diameter_1) > 0.35:
        return landmarks

    diameter = diameter_2

    # take half of the diameter
    radius = diameter / 2

    center = (
        int(
            (abs(most_left_point[0] - most_right_point[0]) / 2)
            + most_left_point[0]
        ),
        int(top_most_point[1] + radius),
    )

    # find the line from landmarks left to apex
    m = (landmarks.apex[1] - landmarks.left[1]) / (
        landmarks.apex[0] - landmarks.left[0]
    )

    radius = radius + radius * config.hip.fem_extention

    if m == 0:
        point_above = (center[0], center[1] + radius)
        point_below = (center[0], center[1] - radius)
    else:
        m_orth = -1 / m
        # equation of orthogonal line: y = m_orth * (x - center[0]) + center[1]
        # solve for x using the circle equation (x-center[0])**2 + (y-center[1])**2 = radius**2

        # break the radius into two parts, based on the
        # sin and cos of the angle between the m_orth and the x-axis
        angle = np.degrees(np.arctan(m_orth))

        # break the radius into two parts, based on the
        # sin and cos of the angle between the m_orth and the x-axis
        radius_x = radius * np.cos(np.radians(angle))
        radius_y = radius * np.sin(np.radians(angle))

        point_above = (
            int(center[0] + radius_x),
            int(center[1] + radius_y),
        )

        point_below = (
            int(center[0] - radius_x),
            int(center[1] - radius_y),
        )

    # ensure that point_above and point_below are the right way round
    # (Larger y represents down)
    if point_above[1] > point_below[1]:
        point_above, point_below = point_below, point_above

    mid_cov_point = smart_find_intersection(
        landmarks.apex,
        landmarks.left,
        point_above,
        point_below,
    )

    landmarks.point_d = point_above
    landmarks.point_D = point_below
    landmarks.mid_cov_point = mid_cov_point

    return landmarks


@warning_decorator(alpha=True)
def find_coverage(landmarks: LandmarksUS) -> float:
    """
    Calculate the Coverage metric.

    :param landmarks: LandmarksUS: The landmarks object.

    :return: float: The Coverage metric.
    """
    if not (
        landmarks
        and landmarks.mid_cov_point
        and landmarks.point_D
        and landmarks.point_d
    ):
        return 0

    coverage = abs(landmarks.mid_cov_point[1] - landmarks.point_D[1]) / abs(
        landmarks.point_D[1] - landmarks.point_d[1]
    )

    # if the mid_point is above the point_D, then the coverage is 0
    if landmarks.mid_cov_point[1] > landmarks.point_D[1]:
        coverage = 0

    return round(coverage, 3)


def draw_coverage(hip: HipDataUS, overlay: Overlay, config: Config) -> Overlay:
    """
    Draw the Coverage metric on the Overlay.

    :param hip: HipDataUS: The Hip Data.
    :param overlay: Overlay: The Overlay.
    :param config: Config: The Config.

    :return: Overlay: The updated Overlay.
    """
    coverage = hip.get_metric(MetricUS.COVERAGE)
    if coverage != 0:
        overlay.draw_lines(
            ((hip.landmarks.point_D, hip.landmarks.point_d),),
        )

        xl, yl = hip.landmarks.mid_cov_point

        if config.visuals.display_full_metric_names:
            title = "Coverage"
        else:
            title = "cov"

        overlay.draw_text(
            f"{title}: {coverage:.2f}",
            int(xl + 10),
            int(yl - 30),
            header="h2",
        )

    return overlay


def bad_coverage(hip: HipDataUS) -> bool:
    """
    Check if the Coverage is bad.

    :param hip: HipDataUS: The Hip Data.

    :return: bool: True if the Coverage is bad.
    """

    if (
        hip.get_metric(MetricUS.COVERAGE) < 0.1
        or hip.get_metric(MetricUS.COVERAGE) > 0.9
    ):
        return True

    return False
