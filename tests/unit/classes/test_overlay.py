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

# Import necessary modules
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from retuve.classes.draw import DrawTypes, Overlay


@pytest.fixture
def overlay(config_xray):
    """
    Fixture for an Overlay object.
    """
    return Overlay((100, 100), config_xray)


def test_add_operation(overlay):
    overlay.add_operation(DrawTypes.LINES, (0, 0), (50, 50), fill=(255, 0, 0))
    assert len(overlay.operations[DrawTypes.LINES]) == 1


def test_draw_cross(overlay):
    overlay.draw_cross((50, 50))
    assert len(overlay.operations[DrawTypes.POINTS]) == 2


def test_draw_segmentation(overlay):
    points = [(10, 10), (20, 20), (30, 10)]
    overlay.draw_segmentation(points)
    assert len(overlay.operations[DrawTypes.SEGS]) == 1


def test_draw_box(overlay):
    box = (10, 10, 30, 30)
    overlay.draw_box(box)
    assert len(overlay.operations[DrawTypes.RECTANGLE]) == 1


def test_draw_skeleton(overlay):
    skeleton_points = [(10, 10), (20, 20), (30, 30)]
    overlay.draw_skeleton(skeleton_points)
    assert len(overlay.operations[DrawTypes.POINTS]) == len(skeleton_points)


def test_draw_lines(overlay):
    line_points = [((0, 0), (10, 10)), ((10, 10), (20, 20))]
    overlay.draw_lines(line_points)
    assert len(overlay.operations[DrawTypes.LINES]) == len(line_points)


def test_draw_text(overlay):
    overlay.draw_text("Test Label", 10, 10, header="h1")
    assert len(overlay.operations[DrawTypes.TEXT]) == 1


def test_draw_circle(overlay):
    overlay.draw_circle((50, 50), 10)
    assert len(overlay.operations[DrawTypes.CIRCLE]) == 1


def test_apply_to_image(overlay):
    # Create a blank image to apply overlay on
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = overlay.apply_to_image(image)
    assert result.shape == (100, 100, 3)  # Should return an RGB image


def test_get_nifti_frame(overlay):
    seg_frame_objs = [
        MagicMock(
            empty=False,
            points=[(10, 10), (20, 20), (30, 10)],
            cls=MagicMock(value=1),
        ),
        MagicMock(empty=True),  # This one should be skipped
    ]
    shape = [100, 100]
    result = overlay.get_nifti_frame(seg_frame_objs, shape)
    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)
