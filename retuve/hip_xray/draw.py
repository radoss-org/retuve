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
Drawing code related to hip xray images.
"""

import time
from typing import List

from numpy.typing import NDArray

from retuve.classes.draw import Overlay
from retuve.classes.seg import SegFrameObjects
from retuve.draw import draw_landmarks, resize_data_for_display
from retuve.hip_xray.classes import HipDataXray
from retuve.hip_xray.metrics.ace import draw_ace
from retuve.hip_xray.metrics.ihdi import draw_ihdi
from retuve.hip_xray.metrics.tonnis import draw_tonnis
from retuve.hip_xray.metrics.wiberg import draw_wiberg
from retuve.keyphrases.config import Config
from retuve.logs import log_timings


def draw_hips_xray(
    hip_datas: List[HipDataXray],
    results: List[SegFrameObjects],
    config: Config,
) -> List[NDArray]:
    """
    Draw the hip xray images

    :param hip_datas: The Hip Datas
    :param results: The Segmentation Results
    :param config: The Config

    :return: The Drawn Images as Numpy Arrays
    """
    draw_timings = []
    image_arrays = []

    for hip, seg_frame_objs in zip(hip_datas, results):
        start = time.time()

        final_hip, final_seg_frame_objs, final_image = resize_data_for_display(
            hip, seg_frame_objs
        )

        overlay = Overlay((final_image.shape[0], final_image.shape[1], 3), config)

        # overlay = draw_seg(final_seg_frame_objs, overlay, config)

        overlay = draw_landmarks(final_hip, overlay, override_line_thickness=6)

        overlay = draw_ace(final_hip, overlay, config)
        overlay = draw_wiberg(final_hip, overlay, config)
        overlay = draw_ihdi(final_hip, overlay, config)
        overlay = draw_tonnis(final_hip, overlay, config)

        img = overlay.apply_to_image(final_image)

        image_arrays.append(img)
        draw_timings.append(time.time() - start)

    log_timings(draw_timings, title="Drawing Speed:")

    return image_arrays
