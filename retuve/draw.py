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
General Drawing Functions used in the retuve package.

"""

import copy
from typing import Union

import numpy as np
from PIL import Image, ImageOps

from retuve.classes.draw import Overlay
from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.general import HipDataUS
from retuve.hip_xray.classes import HipDataXray
from retuve.keyphrases.config import Config

TARGET_SIZE = (700, 700)


def draw_landmarks(
    hip: Union[HipDataUS, HipDataXray],
    overlay: Overlay,
    override_line_thickness: int = None,
) -> Overlay:
    """
    Draw the landmarks on the overlay.

    :param hip: The hip data.
    :param overlay: The overlay to draw on.
    :param override_line_thickness: The line thickness to use for the landmarks.

    :return: The overlay with the landmarks drawn.
    """
    if hip.landmarks is None:
        return overlay

    for landmark in hip.landmarks:
        if landmark is None:
            continue

        overlay.draw_cross(
            landmark, override_line_thickness=override_line_thickness
        )

    return overlay


def draw_seg(
    seg_frame_objs: SegFrameObjects, overlay: Overlay, config: Config
) -> Overlay:
    """
    Draw the segmentation on the overlay.

    :param seg_frame_objs: The segmentation frame objects.
    :param overlay: The overlay to draw on.
    :param config: The configuration.

    :return: The overlay with the segmentation drawn.
    """
    for seg_obj in seg_frame_objs:
        if seg_obj.empty:
            continue

        if seg_obj.points is not None and config.visuals.display_segs:
            overlay.draw_segmentation(seg_obj.points)

        if seg_obj.box is not None and config.visuals.display_boxes:
            overlay.draw_box(seg_obj.box)

        if seg_obj.conf is not None and config.visuals.display_boxes:
            label_text = f"{seg_obj.cls.name}: {seg_obj.conf:.2f}"
            overlay.draw_text(label_text, seg_obj.box[0], seg_obj.box[1])

    return overlay


def _calculate_new_coordinates(original_size, target_size, original_coords):
    original_width, original_height = original_size
    target_width, target_height = target_size
    original_x, original_y = original_coords

    # Calculate aspect ratios
    aspect_ratio_original = original_width / original_height

    # Determine new dimensions while maintaining aspect ratio
    if aspect_ratio_original > 1:  # Wider image
        new_width = target_width
        new_height = int(original_height * (new_width / original_width))
    else:  # Taller or square image
        new_height = target_height
        new_width = int(original_width * (new_height / original_height))

    # Calculate scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Calculate padding (if any)
    pad_x = (target_width - new_width) / 2
    pad_y = (target_height - new_height) / 2

    # Adjust the original coordinates
    new_x = original_x * scale_x + pad_x
    new_y = original_y * scale_y + pad_y

    return new_x, new_y


def resize_data_for_display(
    hip: Union[HipDataUS, HipDataXray], seg_frame_objs: SegFrameObjects
):
    final_image = np.array(
        ImageOps.contain(
            Image.fromarray(seg_frame_objs.img),
            TARGET_SIZE,
            method=Image.NEAREST,
        )
    )

    final_hip = copy.deepcopy(hip)
    final_seg_frame_objs = copy.deepcopy(seg_frame_objs)

    if hip.landmarks:
        for name, landmark in final_hip.landmarks.items():
            if landmark is not None:
                x, y = _calculate_new_coordinates(
                    seg_frame_objs.img.shape[:2],
                    final_image.shape[:2],
                    landmark,
                )
                final_hip.landmarks[name] = (x, y)

    for seg_obj in final_seg_frame_objs:
        if seg_obj.points is not None:
            for i, point in enumerate(seg_obj.points):
                x, y = _calculate_new_coordinates(
                    seg_frame_objs.img.shape[:2],
                    final_image.shape[:2],
                    point,
                )
                seg_obj.points[i] = (x, y)

            if seg_obj.midline is None:
                continue

            for i, point in enumerate(seg_obj.midline_moved):
                x, y = _calculate_new_coordinates(
                    seg_frame_objs.img.shape[:2],
                    final_image.shape[:2],
                    point,
                )
                seg_obj.midline_moved[i] = (x, y)

    return final_hip, final_seg_frame_objs, final_image
