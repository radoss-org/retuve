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


import statistics

import numpy as np
import pytest

from retuve.utils import (
    angle_between_lines,
    find_midline_extremes,
    find_perpendicular_point,
    point_above_below,
    point_left_right,
    rmean,
    rotate_in_place,
    warning_decorator,
)


@pytest.mark.parametrize(
    "p1, p2, p3, expected",
    [
        ((837.18, 414.37), (842.32, 334.53), (877.18, 425.61), -1),
        ((238.73, 309.68), (469.53, 284.55), (142.49, 261.19), 1),
        ((238.73, 309.68), (469.53, 284.55), (533.56, 305.24), -1),
        ((1, 1), (5, 5), (2, 3), -1),  # above
        ((5, 5), (1, 1), (2, 3), -1),  # above
        ((1, 1), (5, 5), (3, 2), 1),  # below
        ((5, 5), (1, 1), (3, 2), 1),  # below
        ((1, 5), (5, 1), (2, 3), 1),  # below
        ((5, 1), (1, 5), (2, 3), 1),  # below
        ((1, 5), (5, 1), (3, 4), -1),  # above
        ((5, 1), (1, 5), (3, 4), -1),  # above
        # horizontal line
        ((0, 0), (1, 0), (0.5, 1), -1),
        ((0, 0), (1, 0), (0.5, -1), 1),
        ((0, 0), (1, 0), (0.5, 0), 0),
    ],
)
def test_point_above_below(p1, p2, p3, expected):
    result = point_above_below(p1, p2, p3)
    if expected == 0:
        assert result == 0
    elif expected > 0:
        assert result > 0
    else:
        assert result < 0


@pytest.mark.parametrize(
    "p1, p2, p3, expected",
    [
        ((842.32, 334.53), (837.18, 414.37), (877.18, 425.61), -1),
        ((279.43, 405.93), (284.57, 326.09), (289.92, 419.09), -1),
        ((213.99, 357.05), (205.33, 277.52), (142.49, 261.19), 1),
        ((529.9, 323.08), (521.25, 243.55), (533.56, 305.24), -1),
        ((521.25, 243.55), (529.9, 323.08), (533.56, 305.24), -1),
        ((1, 1), (5, 5), (2, 3), 1),  # left
        ((5, 5), (1, 1), (2, 3), 1),  # left
        ((1, 1), (5, 5), (3, 2), -1),  # right
        ((5, 5), (1, 1), (3, 2), -1),  # right
        ((1, 5), (5, 1), (2, 3), 1),  # left
        ((5, 1), (1, 5), (2, 3), 1),  # left
        ((1, 5), (5, 1), (3, 4), -1),  # right
        ((5, 1), (1, 5), (3, 4), -1),  # right
        # vertical line
        ((0, 0), (0, 1), (1, 0.5), -1),
        ((0, 0), (0, 1), (-1, 0.5), 1),
        ((0, 0), (0, 1), (0, 0.5), 0),
        ((400.3368238, 484.79930097), (406, 405), (400, 537), -1),
    ],
)
def test_point_left_right(p1, p2, p3, expected):
    result = point_left_right(p1, p2, p3)
    if expected == 0:
        assert result == 0
    elif expected > 0:
        assert result > 0
    else:
        assert result < 0


def test_rmean():
    """Test the rmean function."""
    # Test with simple values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rmean(values)
    assert result == 3.0, f"Expected 3.0, got {result}"
    assert isinstance(result, float), "Result should be a float"

    # Test with decimal values
    values = [1.234, 2.345, 3.456]
    result = rmean(values)
    expected = round((1.234 + 2.345 + 3.456) / 3, 2)
    assert result == expected, f"Expected {expected}, got {result}"

    # Test with single value
    values = [42.0]
    result = rmean(values)
    assert result == 42.0, f"Expected 42.0, got {result}"

    # Test with empty list
    with pytest.raises(statistics.StatisticsError):
        rmean([])


def test_find_midline_extremes():
    """Test the find_midline_extremes function."""
    # Test with valid midline data
    midline = np.array([[10, 5], [20, 3], [30, 8], [15, 1], [25, 10]])
    left_most, right_most = find_midline_extremes(midline)

    assert left_most is not None, "Left most should not be None"
    assert right_most is not None, "Right most should not be None"
    assert left_most[1] == 1, f"Left most should have y=1, got {left_most[1]}"
    assert right_most[1] == 10, f"Right most should have y=10, got {right_most[1]}"

    # Test with empty midline
    empty_midline = np.array([])
    left_most, right_most = find_midline_extremes(empty_midline)
    assert left_most is None, "Left most should be None for empty midline"
    assert right_most is None, "Right most should be None for empty midline"


def test_angle_between_lines():
    """Test the angle_between_lines function."""
    # Test with perpendicular lines
    line1 = ((0, 0), (1, 0))  # Horizontal line
    line2 = ((0, 0), (0, 1))  # Vertical line
    angle = angle_between_lines(line1, line2)
    assert abs(angle - 90.0) < 0.001, f"Expected 90 degrees, got {angle}"

    # Test with parallel lines
    line1 = ((0, 0), (1, 0))  # Horizontal line
    line2 = ((0, 1), (1, 1))  # Parallel horizontal line
    angle = angle_between_lines(line1, line2)
    assert abs(angle - 0.0) < 0.001, f"Expected 0 degrees, got {angle}"

    # Test with 45-degree angle
    line1 = ((0, 0), (1, 0))  # Horizontal line
    line2 = ((0, 0), (1, 1))  # 45-degree line
    angle = angle_between_lines(line1, line2)
    assert abs(angle - 45.0) < 0.001, f"Expected 45 degrees, got {angle}"

    # Test with same line
    line1 = ((0, 0), (1, 1))
    line2 = ((0, 0), (1, 1))
    angle = angle_between_lines(line1, line2)
    assert abs(angle - 0.0) < 0.001, f"Expected 0 degrees, got {angle}"


def test_find_perpendicular_point():
    """Test the find_perpendicular_point function."""
    # Test with horizontal line
    p1 = (0, 0)
    p2 = (10, 0)
    p3 = (5, 5)
    distance = 10

    p4 = find_perpendicular_point(p1, p2, p3, distance)

    assert isinstance(p4, np.ndarray), "Result should be numpy array"
    assert p4.shape == (2,), "Result should have 2 dimensions"

    # Check that the line p3-p4 is perpendicular to p1-p2
    vec1 = np.array(p2) - np.array(p1)  # Original line vector
    vec2 = p4 - np.array(p3)  # Perpendicular line vector
    dot_product = np.dot(vec1, vec2)
    assert abs(dot_product) < 0.001, "Lines should be perpendicular"

    # Check distance
    actual_distance = np.linalg.norm(p4 - np.array(p3))
    assert (
        abs(actual_distance - distance) < 0.001
    ), f"Distance should be {distance}, got {actual_distance}"

    # Test with vertical line
    p1 = (0, 0)
    p2 = (0, 10)
    p3 = (5, 5)
    distance = 5

    p4 = find_perpendicular_point(p1, p2, p3, distance)

    # Check perpendicularity
    vec1 = np.array(p2) - np.array(p1)
    vec2 = p4 - np.array(p3)
    dot_product = np.dot(vec1, vec2)
    assert abs(dot_product) < 0.001, "Lines should be perpendicular"

    # Test with diagonal line
    p1 = (0, 0)
    p2 = (10, 10)
    p3 = (5, 5)
    distance = 7.071  # sqrt(50)

    p4 = find_perpendicular_point(p1, p2, p3, distance)

    # Check perpendicularity
    vec1 = np.array(p2) - np.array(p1)
    vec2 = p4 - np.array(p3)
    dot_product = np.dot(vec1, vec2)
    assert abs(dot_product) < 0.001, "Lines should be perpendicular"


def test_rotate_in_place():
    """Test the rotate_in_place function."""
    # Test 90-degree rotation
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = 90

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)

    assert rotated_p1 == p1, "Center point should not change"
    assert isinstance(rotated_p2, np.ndarray), "Rotated point should be numpy array"

    # Check that rotated point is at (0, 1) approximately
    assert abs(rotated_p2[0] - 0) < 0.001, f"Expected x=0, got {rotated_p2[0]}"
    assert abs(rotated_p2[1] - 1) < 0.001, f"Expected y=1, got {rotated_p2[1]}"

    # Test 180-degree rotation
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = 180

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)

    # Check that rotated point is at (-1, 0) approximately
    assert abs(rotated_p2[0] - (-1)) < 0.001, f"Expected x=-1, got {rotated_p2[0]}"
    assert abs(rotated_p2[1] - 0) < 0.001, f"Expected y=0, got {rotated_p2[1]}"

    # Test 45-degree rotation
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = 45

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)

    # Check that rotated point is at (cos(45), sin(45)) approximately
    expected_x = np.cos(np.radians(45))
    expected_y = np.sin(np.radians(45))
    assert (
        abs(rotated_p2[0] - expected_x) < 0.001
    ), f"Expected x={expected_x}, got {rotated_p2[0]}"
    assert (
        abs(rotated_p2[1] - expected_y) < 0.001
    ), f"Expected y={expected_y}, got {rotated_p2[1]}"

    # Test with non-origin center
    p1 = (5, 5)
    p2 = (6, 5)
    angle_deg = 90

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)

    assert rotated_p1 == p1, "Center point should not change"
    # Check that rotated point is at (5, 6) approximately
    assert abs(rotated_p2[0] - 5) < 0.001, f"Expected x=5, got {rotated_p2[0]}"
    assert abs(rotated_p2[1] - 6) < 0.001, f"Expected y=6, got {rotated_p2[1]}"


def test_warning_decorator():
    """Test the warning_decorator function."""

    # Test alpha warning
    @warning_decorator(alpha=True)
    def test_func_alpha():
        return "alpha_result"

    # Test beta warning
    @warning_decorator(beta=True, paper_url="https://example.com")
    def test_func_beta():
        return "beta_result"

    # Test validated warning
    @warning_decorator(validated=True)
    def test_func_validated():
        return "validated_result"

    # Test that functions still work
    assert test_func_alpha() == "alpha_result"
    assert test_func_beta() == "beta_result"
    assert test_func_validated() == "validated_result"

    # Test that warnings are only shown once per function
    # The second call should not show warnings again
    test_func_alpha()
    test_func_beta()
    test_func_validated()


def test_warning_decorator_with_paper_url():
    """Test the warning_decorator with paper URL."""

    @warning_decorator(beta=True, paper_url="https://test-paper.com")
    def test_func_with_url():
        return "result"

    # Function should work normally
    assert test_func_with_url() == "result"


def test_find_perpendicular_point_edge_cases():
    """Test find_perpendicular_point with edge cases."""
    # Test with zero distance
    p1 = (0, 0)
    p2 = (10, 0)
    p3 = (5, 5)
    distance = 0

    p4 = find_perpendicular_point(p1, p2, p3, distance)
    assert np.allclose(
        p4, np.array(p3)
    ), "With zero distance, should return original point"

    # Test with very small distance
    distance = 0.001
    p4 = find_perpendicular_point(p1, p2, p3, distance)
    actual_distance = np.linalg.norm(p4 - np.array(p3))
    assert (
        abs(actual_distance - distance) < 0.001
    ), f"Distance should be {distance}, got {actual_distance}"


def test_rotate_in_place_edge_cases():
    """Test rotate_in_place with edge cases."""
    # Test with zero angle
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = 0

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)
    assert np.allclose(
        rotated_p2, np.array(p2)
    ), "With zero angle, should return original point"

    # Test with 360-degree rotation
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = 360

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)
    assert np.allclose(
        rotated_p2, np.array(p2)
    ), "With 360 degrees, should return original point"

    # Test with negative angle
    p1 = (0, 0)
    p2 = (1, 0)
    angle_deg = -90

    rotated_p1, rotated_p2 = rotate_in_place(p1, p2, angle_deg)
    # Should be equivalent to 270-degree rotation
    assert abs(rotated_p2[0] - 0) < 0.001, f"Expected x=0, got {rotated_p2[0]}"
    assert abs(rotated_p2[1] - (-1)) < 0.001, f"Expected y=-1, got {rotated_p2[1]}"
