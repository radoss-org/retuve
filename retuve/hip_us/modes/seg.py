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
All code relating to going from segmentation results to landmarks is in this file.
"""

import copy
import time
from typing import List

import cv2
import numpy as np
from skimage.morphology import skeletonize

from retuve.classes.seg import (
    MidLine,
    NDArrayImg_NxNx3_AllWhite,
    SegFrameObjects,
    SegObject,
)
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.hip_us.classes.general import LandmarksUS
from retuve.hip_us.handlers.segs import remove_bad_objs
from retuve.hip_us.metrics.alpha import find_alpha_landmarks
from retuve.hip_us.metrics.coverage import find_cov_landmarks
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import MidLineMove
from retuve.logs import log_timings
from retuve.utils import find_midline_extremes


def get_midlines(
    mask: NDArrayImg_NxNx3_AllWhite, config: Config, color: int = 255
) -> MidLine:
    """
    Get the midline of the illium mask. This is used to '
    find the landmarks for the alpha angle.

    :param mask: The mask of the illium.
    :param config: The configuration object.
    :param color: The color of the midline.

    :return: The midline of the illium.
    """

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # Takes bool as input and returns bool as output
    midline = skeletonize(mask > 0)
    midline = midline.astype(np.uint8) * color

    midline_moved = midline

    # move the skeleton up and right
    if config.hip.midline_move_method == MidLineMove.BASIC:
        height, width = midline_moved.shape
        midline_moved = np.roll(midline_moved, -1 * int(height / 51), axis=0)
        midline_moved = np.roll(midline_moved, int(width / 86), axis=1)

    for midl in [midline_moved, midline]:
        left_most, right_most = find_midline_extremes(
            np.column_stack(np.where(midl == color))
        )
        if left_most is not None and right_most is not None:
            width = right_most[1] - left_most[1]

            new_left_bound = max(int(left_most[1] + width * 0.01), 4)
            new_right_bound = max(int(right_most[1] - width * 0.01), 4)

            # Apply the boundary conditions to remove edges
            midl[:, :new_left_bound] = 0
            midl[:, new_right_bound:] = 0

    midline = np.column_stack(np.where(midline == color))
    midline_moved = np.column_stack(np.where(midline_moved == color))

    return midline, midline_moved


def segs_2_landmarks_us(
    results: List[SegFrameObjects], config: Config
) -> List[LandmarksUS]:
    """
    Converts a list of segmentation results to a list of landmarks.

    :param results: A list of segmentation results.
    :param config: The configuration object.

    :return: A list of landmarks.
    """
    hip_landmarks = []
    timings = []
    hip_objs_list = []
    fem_head_ilium_wrong_way_round = 0
    all_rejection_reasons = []

    for seg_frame_objs in results:

        if all(seg_obj.empty for seg_obj in seg_frame_objs):
            hip_objs_list.append(None)
            all_rejection_reasons.append([])
            continue

        hip_objs = {
            HipLabelsUS.IlliumAndAcetabulum: [],
            HipLabelsUS.FemoralHead: [],
            HipLabelsUS.OsIchium: [],
        }

        for seg_obj in seg_frame_objs:
            hip_objs[seg_obj.cls].append(seg_obj)

        hip_objs, wrong_way_round, rejection_reasons = remove_bad_objs(
            hip_objs, seg_frame_objs.img
        )
        if wrong_way_round:
            fem_head_ilium_wrong_way_round += 1

        hip_objs_list.append(hip_objs)
        all_rejection_reasons.append(rejection_reasons)

    if fem_head_ilium_wrong_way_round > 5 and config.hip.allow_horizontal_flipping:
        # Flip images
        for seg_frame_objs in results:
            seg_frame_objs.img = cv2.flip(seg_frame_objs.img, 1)

        for hip_objs in hip_objs_list:
            if hip_objs is None:
                continue

            for seg_obj in hip_objs.values():
                seg_obj.flip_horizontally(results[0].img.shape[1])

            ilium = hip_objs.get(HipLabelsUS.IlliumAndAcetabulum, None)

            if (
                ilium is not None
                and ilium.box is not None
                and ilium.box[0] > results[0].img.shape[1] / 2
            ):
                hip_objs[HipLabelsUS.IlliumAndAcetabulum] = SegObject(empty=True)

    for seg_frame_objs, hip_objs in zip(results, hip_objs_list):
        start = time.time()

        landmarks = LandmarksUS()
        if hip_objs is None:
            hip_landmarks.append(landmarks)
            continue

        # set the seg_frame_objs to only have the good seg_objs
        seg_frame_objs.seg_objects = [
            obj for obj in list(hip_objs.values()) if obj is not None
        ]

        illium = hip_objs.get(HipLabelsUS.IlliumAndAcetabulum, None)
        femoral = hip_objs.get(HipLabelsUS.FemoralHead, None)

        landmarks = find_alpha_landmarks(illium, landmarks, config)
        landmarks = find_cov_landmarks(femoral, landmarks, config)

        hip_landmarks.append(landmarks)
        timings.append(time.time() - start)

    log_timings(timings, title="Seg->Landmarking Speed:")
    return hip_landmarks, all_rejection_reasons


def pre_process_segs_us(
    results: List[SegFrameObjects], config: Config
) -> tuple[List[SegFrameObjects], tuple]:
    """
    Pre-processes the segmentation results for the hip US module.

    :param results: A list of segmentation results.
    :param config: The configuration object.

    :return: A tuple containing the pre-processed
             segmentation results and the shape of the image.
    """
    timings = []
    for frame_seg_objs in results:
        start = time.time()
        for seg_object in frame_seg_objs:
            if seg_object.empty or seg_object.cls != HipLabelsUS.IlliumAndAcetabulum:
                continue

            seg_object.midline, seg_object.midline_moved = get_midlines(
                seg_object.mask, config
            )

        timings.append(time.time() - start)

    log_timings(timings, title="Skeletonization Speed:")

    img = results[0].img
    shape = img.shape

    return results, shape
