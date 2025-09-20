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
Handles bad Hip Data Objects by removing outliers and empty frames.
"""

from typing import List

import numpy as np
from retuve.hip_us.classes.general import HipDatasUS, HipDataUS
from retuve.hip_us.metrics.alpha import bad_alpha
from retuve.hip_us.metrics.coverage import bad_coverage
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import HipMode


def remove_outliers(hip_datas: HipDatasUS, config: Config) -> List[bool]:
    """
    Remove outliers from the HipDatasUS object.

    :param hip_datas: HipDatasUS object.
    :param config: Config object.

    :return: List of booleans indicating which frames to keep.
    """
    pred_made = [(True if hip.marked() else False) for hip in hip_datas]

    # Use this as a sliding window to find the position where
    # the most Trues fit in the window
    # Get total number of True values
    total_true = sum(pred_made)

    # Sliding window to find the position where the most Trues fit in the window
    max_true = 0
    max_true_index = 0

    for i in range(len(pred_made) - total_true + 1):
        current_window_true = sum(pred_made[i : i + total_true])
        if current_window_true > max_true:
            max_true = current_window_true
            max_true_index = i

    # Create a list of Trues and Falses
    keep = [False] * len(pred_made)
    for i in range(max_true_index, max_true_index + total_true):
        if pred_made[i]:
            keep[i] = True

    return keep


def left_apex_line_flat(hip: HipDataUS) -> bool:
    """
    Check if the left apex line is flat.

    :param hip: HipDataUS object.

    :return: Boolean indicating if the left apex line is flat.
    """
    if hip.landmarks.left is None or hip.landmarks.apex is None:
        return True

    C, A, B = (
        np.array(hip.landmarks.left),
        np.array(hip.landmarks.apex),
        np.array((hip.landmarks.apex[0], hip.landmarks.left[1])),
    )

    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)

    angle = np.arccos((a**2 + b**2 - np.linalg.norm(A - B) ** 2) / (2 * a * b))
    angle = np.degrees(angle)

    return abs(angle) < 10


def apex_right_points_too_close(hip: HipDataUS) -> bool:
    """
    Check if the apex and right points are too close.

    :param hip: HipDataUS object.

    :return: Boolean indicating if the apex and right points are too close.
    """
    if hip.landmarks.right is None or hip.landmarks.apex is None:
        return True

    return (
        np.linalg.norm(np.array(hip.landmarks.right) - np.array(hip.landmarks.apex))
        < 30
    )


def handle_bad_frames(hip_datas: HipDatasUS, config: Config) -> HipDatasUS:
    """
    Handle bad frames by removing outliers and empty frames.

    :param hip_datas: HipDatasUS object.
    :param config: Config object.

    :return: HipDatasUS object.
    """

    if config.batch.hip_mode == HipMode.US2DSW:
        keep = [(True if hip.marked() else False) for hip in hip_datas]
    else:
        keep = remove_outliers(hip_datas, config)

    bad_frame_reasons = {}

    for i, (hip, rejection_reasons) in enumerate(
        zip(hip_datas, hip_datas.all_seg_rejection_reasons)
    ):

        empty_hip = HipDataUS(
            frame_no=hip.frame_no,
        )

        if not keep[i]:
            hip_datas[i] = empty_hip
            if config.batch.hip_mode == HipMode.US2DSW:
                bad_frame_reasons[i] = " ".join(rejection_reasons)
                if not rejection_reasons:
                    bad_frame_reasons[i] = "Not Enough Data"
            continue

        if (not hip.metrics) or all(metric.value == 0 for metric in hip.metrics):
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "No Metrics"
            continue

        if hip.landmarks is None:
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "No Landmarks"
            continue

        if bad_alpha(hip):
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "Alpha Angle Non-Sensical"
            continue

        if not left_apex_line_flat(hip) and not config.hip.allow_irregular_illiums:
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "Ilium Line not Flat"
            continue

        if bad_coverage(hip, config):
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "Coverage Value Non-Sensical"
            continue

        if apex_right_points_too_close(hip):
            hip_datas[i] = empty_hip
            bad_frame_reasons[i] = "Apex and Right Too Close"
            continue

    hip_datas.bad_frame_reasons = bad_frame_reasons
    return hip_datas
