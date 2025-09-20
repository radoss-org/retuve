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
Utility functions for the Retuve package.
"""

import functools
import logging
import os
import statistics
import sys
from typing import List, Tuple

import numpy as np

from retuve.logs import ulogger
from retuve.typehints import MidLine

if getattr(sys, "frozen", False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    RETUVE_DIR = sys._MEIPASS
else:
    RETUVE_DIR = os.path.dirname(os.path.realpath(__file__))


def register_config_dirs(config, other_dirs=[]):
    hippa_log_file_dir = os.path.dirname(config.api.hippa_logging_file)
    sql_path = os.path.dirname(config.api.db_path)
    dirs = [
        config.api.savedir,
        config.api.upload_dir,
        hippa_log_file_dir,
        sql_path,
    ]

    dirs.extend(other_dirs)
    dirs.extend(config.trak.datasets)
    dirs.extend(config.batch.datasets)

    for dir in dirs:
        if dir and not os.path.exists(dir):
            ulogger.info(f"Creating directory: {dir}")
            os.makedirs(dir, exist_ok=True)


def rmean(values: List[float]) -> float:
    """
    Calculate the rounded mean of a list of values.

    :param values: The list of values to calculate the mean of.
    :return: The rounded mean of the values.
    """
    return round(statistics.mean(values), 2)


def find_midline_extremes(
    midline: MidLine,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Finds the left-most and right-most white pixels in the midline.

    :param midline: The midline to find the extremes of.
    :return: The left-most and right-most white pixels.
    """
    # Identifying the left-most and right-most white pixels
    if midline.size > 0:
        left_most = midline[midline[:, 1].argmin()]
        right_most = midline[midline[:, 1].argmax()]
        return left_most, right_most
    else:
        logging.warning("No white pixels found in the image.")
        return None, None


def angle_between_lines(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]],
):
    """
    Calculate the angle between two lines.

    :param line1: The first line.
    :param line2: The second line.
    :return: The angle between the two lines.
    """
    # convert to numpy arrays
    line1 = np.array(line1)
    line2 = np.array(line2)

    v1 = line1[1] - line1[0]
    v2 = line2[1] - line2[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def find_perpendicular_point(p1, p2, p3, distance=80):
    """
    Finds a 4th point (p4) such that the line (p3, p4) is perpendicular
    to the line (p1, p2), with p4 being a specific distance below p3.

    Args:
      p1 (tuple | list | np.ndarray): The first point of the original line.
      p2 (tuple | list | np.ndarray): The second point of the original line.
      p3 (tuple | list | np.ndarray): The starting point of the new line.
      distance (float): The desired distance between p3 and p4.

    Returns:
      np.ndarray: The calculated fourth point (p4).
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    dir_vector = p2 - p1
    perp_vector = np.array([-dir_vector[1], dir_vector[0]])

    norm = np.linalg.norm(perp_vector)
    unit_perp_vector = perp_vector / norm

    if unit_perp_vector[1] < 0:
        unit_perp_vector *= -1

    offset = unit_perp_vector * distance
    p4 = p3 + offset

    return p4


def rotate_in_place(p1, p2, angle_deg=45):
    """
    Rotates a point around another point by a given angle.

    :param p1: The point to rotate around.
    :param p2: The point to rotate.
    :param angle_deg: The angle to rotate by.
    :return: The rotated point.
    """
    v = np.array(p2) - np.array(p1)
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p2_rot = np.array(p1) + rot @ v
    return p1, p2_rot


def point_above_below(p1, p2, p3):
    """
    Returns:
        >0 if p3 is above the line from p1 to p2,
        <0 if p3 is below,
        0 if p3 is exactly on the line.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if x2 == x1:
        if y3 == y1:
            return 0
        return y1 - y3
    m = (y2 - y1) / (x2 - x1)
    y_on_line = m * (x3 - x1) + y1
    if y3 == y_on_line:
        return 0

    return y_on_line - y3


def point_left_right(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculates if a point p3 is to the left or right of a line segment p1-p2.
    The function has specific behavior:

    1. If p3 is horizontally outside the segment, it returns the horizontal
    distance to the nearest endpoint's x-coordinate.
    2. If p3 is horizontally within the segment, it returns the vertical
    distance to the line, with the sign indicating left/right.

    Note: This complex approach is needed instead of a simple cross product because:
    - We need consistent left/right results regardless of p1/p2 order
    - We want meaningful results for points near (but outside) the segment ends
    - The extension creates a "tolerance zone" for practical applications like UI hit testing

    :param p1: The first point of the line segment (x1, y1).
    :param p2: The second point of the line segment (x2, y2).
    :param p3: The point to test (x3, y3).
    :return:
    - A positive value for "left".
    - A negative value for "right".
    - Zero if the point is on the line.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Extend the line segment by 1000 pixels in both directions
    if x1 == x2:
        # Vertical line case - extend in y direction instead
        extension = 1000
        if y2 > y1:
            extended_y1 = y1 - extension
            extended_y2 = y2 + extension
        else:
            extended_y1 = y1 + extension
            extended_y2 = y2 - extension
        extended_x1, extended_x2 = x1, x2
    else:
        # Calculate slope and extend in x direction
        slope = (y2 - y1) / (x2 - x1)
        extension = 1000

        # Extend leftward from the leftmost point
        if x1 < x2:
            extended_x1 = x1 - extension
            extended_y1 = y1 - slope * extension
            extended_x2 = x2 + extension
            extended_y2 = y2 + slope * extension
        else:
            extended_x1 = x1 + extension
            extended_y1 = y1 + slope * extension
            extended_x2 = x2 - extension
            extended_y2 = y2 - slope * extension

    # Use extended points for the rest of the calculation
    x1, y1 = extended_x1, extended_y1
    x2, y2 = extended_x2, extended_y2

    # Part 1: Handle points outside the x-range of the segment p1-p2.
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    if x3 < min_x:
        return min_x - x3
    if x3 > max_x:
        return max_x - x3

    # Part 2: Ensure p1 is the left-most point for consistent calculations.
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1

    if x1 == x2:
        return 0.0

    slope = (y2 - y1) / (x2 - x1)
    y_on_line = slope * (x3 - x1) + y1

    vertical_diff = y3 - y_on_line

    if y1 == y2:
        return vertical_diff

    sign_multiplier = 1.0 if y1 < y2 else -1.0
    return vertical_diff * sign_multiplier


# Global registry to track which functions have shown warnings
_warned_functions = set()


def warning_decorator(alpha=False, beta=False, validated=False, paper_url=None):
    """
    Decorator to print warning messages when decorated functions are run, based on specified flags.
    Each warning is only printed once per function per program run.

    Args:
        alpha (bool): If True, prints an alpha warning message. Defaults to False.
        beta (bool): If True, prints a beta warning message. Defaults to False.
        validated (bool): If True, prints a validated warning message. Defaults to False.
        paper_url (str, optional): URL of the paper related to the beta validation. Defaults to None.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)  # Use function's id as unique identifier

            # Only show warning if this function hasn't been warned about yet
            if func_id not in _warned_functions:
                if alpha:
                    message = (
                        f"\nWARNING: {func.__name__} is in early experimentation "
                        "and should not be used for anything outside of "
                        "research by an expert in the field of Hip Dysplasia "
                        "AI. Use it with extreme caution."
                    )
                elif beta:
                    message = (
                        f"\nWARNING: {func.__name__} has limited validation, with at "
                        "least one paper released on a small scale, with "
                        "accepted Inter-class Correlation coefficients (ICC). "
                        "Use with caution, ensuring to read the full paper."
                    )
                    print(message)
                elif validated:
                    message = f"\nWARNING: Using academic-only validated function {func.__name__} - use with caution, and never in a clinical setting."

                if paper_url and message:
                    message += f"  Paper URL: {paper_url}"

                if message:
                    print(message)

                # Add this function to the set of warned functions
                _warned_functions.add(func_id)

            return func(*args, **kwargs)

        return wrapper

    return decorator
