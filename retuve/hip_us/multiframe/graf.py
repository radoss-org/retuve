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
High Level Functions for Multiframe Analysis
"""

import json
import os
from typing import List

import cv2
import numpy as np
from filelock import FileLock

from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.hip_us.classes.general import HipDatasUS
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import GrafSelectionMethod, MetricUS
from retuve.logs import ulogger
from retuve.utils import warning_decorator


def _get_left_apex_angle(hip) -> bool:
    """
    Check if the left apex line is flat.

    :param hip: HipDataUS object.

    :return: Boolean indicating if the left apex line is flat.
    """
    if hip.landmarks is None:
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

    return int(angle)


def _get_os_ichium_area(seg_frame_objs) -> float:
    os_ichium = [
        seg_obj for seg_obj in seg_frame_objs if seg_obj.cls == HipLabelsUS.OsIchium
    ]
    os_ishium_area = 0
    # Gives a value of 0 or 2
    if len(os_ichium) != 0:
        os_ishium_area = round(os_ichium[0].area(), 1)

    return os_ishium_area


def _get_femoral_head_area(seg_frame_objs) -> float:
    femoral_head = [
        seg_obj for seg_obj in seg_frame_objs if seg_obj.cls == HipLabelsUS.FemoralHead
    ]
    femoral_head_area = 0
    if len(femoral_head) != 0:
        femoral_head_area = round(femoral_head[0].area(), 1)

    return femoral_head_area


def _get_apex_right_distance(hip) -> float:
    apex_right_distance = 0
    if hip.landmarks:
        apex_right_distance = abs(hip.landmarks.apex[1] - hip.landmarks.right[1])

    return apex_right_distance


def _get_femoral_head_roundness(seg_frame_objs) -> float:
    femoral_head = [
        seg_obj for seg_obj in seg_frame_objs if seg_obj.cls == HipLabelsUS.FemoralHead
    ]
    roundness_ratio = 0
    if len(femoral_head) != 0:
        foreground_mask = (
            np.all(femoral_head[0].mask == [255, 255, 255], axis=-1).astype(np.uint8)
            * 255
        )

        # Step 2: Detect edges
        edges = cv2.Canny(foreground_mask, 50, 150)

        # Step 3: Fit a circle to the detected edges
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Get the largest contour (assuming it's the femoral head)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a minimum enclosing circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            circle_area = np.pi * (radius**2)
            contour_area = cv2.contourArea(largest_contour)

            # Step 4: Calculate roundness
            # Roundness ratio of contour area to enclosing circle area (closer to 1 is more round)
            roundness_ratio = contour_area / circle_area if circle_area != 0 else 0

    return roundness_ratio


def graf_frame_algo(
    hip_zipped_data,
    max_alpha,
    first_illium_frame,
    last_illium_frame,
    file_id=None,
    config: Config = None,
):
    """
    Get the Graf Frame for the hip US module as a weighted
    mix of the max alpha value and flatness of the angle.

    See an analysis of how these weights were found here: https://files.mcaq.me/xj3kb.png.

    The code will be included directly in Retuve in the future update.

    :param hip_zipped_data: The hip data and results.
    :param max_alpha: The Max Alpha
    :param first_illium_frame: The first illium frame.
    :param last_illium_frame: The last illium frame.

    :return: Weighted score based on alpha and flatness.
    """

    alpha_weight = 5.6  # Largest Alpha Angle
    line_flatness_weight = 4.71  # Flatness of illium
    os_ishium_weight = 14.51  # Os Ichium Area
    femoral_head_weight = 12.81  # Femoral Head Area
    apex_right_distance_weight = 0.96  # Distance between apex and right
    femoral_head_roundness_weight = 1.42  # Roundness of femoral head
    graf_frame_position_weight = 0.16  # Position of the frame in the illium

    hip_data, seg_frame_objs = hip_zipped_data

    # Gives values between 1 and 7
    alpha_normalisation = max_alpha / 4
    alpha_value = round(hip_data.get_metric(MetricUS.ALPHA) / alpha_normalisation, 2)

    line_flattness_normalisation = 2
    # Gives values varying between 0 and 10
    line_flatness_value = (
        10 - _get_left_apex_angle(hip_data)
    ) / line_flattness_normalisation

    # Print the image area
    image_area = seg_frame_objs.img.shape[0] * seg_frame_objs.img.shape[1]

    os_ichium_normalisation = image_area / 140
    os_ichium_value = _get_os_ichium_area(seg_frame_objs) / os_ichium_normalisation

    # do the same thing for the femoral head
    femoral_head_normalisation = image_area / 14
    femoral_head_value = (
        _get_femoral_head_area(seg_frame_objs) / femoral_head_normalisation
    )

    apex_right_distance_normalisation = seg_frame_objs.img.shape[0] / 20
    apex_right_distance_value = (
        _get_apex_right_distance(hip_data) / apex_right_distance_normalisation
    )

    # Weigh being in the middle third of the illium frames
    illium_third_length = (last_illium_frame - first_illium_frame) // 3
    middle_third_range = range(
        first_illium_frame + illium_third_length,
        last_illium_frame - illium_third_length,
    )

    graf_frame_position_value = 0
    if hip_data.frame_no in middle_third_range:
        graf_frame_position_value = 3

    femoral_head_roundness_normalisation = 2
    femoral_head_roundness_value = (
        _get_femoral_head_roundness(seg_frame_objs)
        * femoral_head_roundness_normalisation
    )

    if config.hip.allow_irregular_illiums:
        line_flatness_value = 0
        line_flatness_weight = 0

    # Calculate the weighted score based on alpha and angle flatness
    final_score = round(
        (
            alpha_weight * alpha_value
            + line_flatness_weight * line_flatness_value
            + os_ishium_weight * os_ichium_value
            + femoral_head_weight * femoral_head_value
            + apex_right_distance_weight * apex_right_distance_value
            + graf_frame_position_weight * graf_frame_position_value
            + femoral_head_roundness_weight * femoral_head_roundness_value
        ),
        2,
    )

    if file_id and getattr(config, "do_calibration", False):
        data = {
            "alpha_value": alpha_value,
            "line_flatness_value": line_flatness_value,
            "os_ichium_value": os_ichium_value,
            "femoral_head_value": femoral_head_value,
            "apex_right_distance_value": apex_right_distance_value,
            "graf_frame_position_value": graf_frame_position_value,
            "femoral_head_roundness_value": femoral_head_roundness_value,
        }

        # Create folder for the JSON file
        json_folder = f"./env/calibration/data"
        os.makedirs(json_folder, exist_ok=True)

        # Define the path to the JSON file and the lock file
        json_file_path = f"{json_folder}/{file_id.replace('.dcm', '.json')}"
        lock_file_path = f"{json_folder}/{file_id.replace('.dcm', '.lock')}"

        # Use a file lock to prevent concurrent access issues
        lock = FileLock(lock_file_path)

        # Acquire the lock before reading or writing
        with lock:
            # If the JSON file already exists, load its contents; otherwise, start with an empty dictionary
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    hip_data_dict = json.load(f)
            else:
                hip_data_dict = {}

            # Add or update the data for this frame_no
            hip_data_dict[hip_data.frame_no] = data

            # Save the updated data back to the JSON file
            with open(json_file_path, "w") as f:
                json.dump(hip_data_dict, f, indent=4)

    return final_score


@warning_decorator(alpha=True)
def find_graf_plane(
    hip_datas: HipDatasUS, results: List[SegFrameObjects], config: Config
) -> HipDatasUS:
    """
    Find the Graf Plane for the hip US module.

    :param hip_datas: The hip data.
    :param results: The results of the segmentation.
    :param config: The configuration.

    :return: The hip data with the Graf Plane.
    """

    if config.hip.graf_selection_method == GrafSelectionMethod.MANUAL_FRAME:
        if config.hip.graf_frame_selection is None:
            raise ValueError(
                "Graf Frame Selection is None. "
                "Please set config.hip.graf_frame_selection to a valid frame number."
            )

        graf_frame = config.hip.graf_frame_selection

        hip_datas.graf_frame = graf_frame

        hip_datas.grafs_hip = [hip for hip in hip_datas if hip.frame_no == graf_frame][
            0
        ]

        # build the graf confs
        # 0 for each frame except the graf frame
        hip_datas.graf_confs = [
            0 if hip_data.frame_no != graf_frame else 1
            for hip_data in hip_datas.hip_datas
        ]

        return hip_datas

    if config.hip.graf_selection_method == GrafSelectionMethod.MANUAL_FEATURES:
        return find_graf_plane_manual_features(hip_datas, results, config)

    elif config.hip.graf_selection_method == GrafSelectionMethod.OBJ_CLASSIFY:
        raise NotImplementedError(
            f"Unsupported Graf Selection Method: {config.hip.graf_selection_method}"
        )

    else:
        raise NotImplementedError(
            f"Unsupported Graf Selection Method: {config.hip.graf_selection_method}"
        )


def find_graf_plane_manual_features(
    hip_datas: HipDatasUS, results: List[SegFrameObjects], config: Config
) -> HipDatasUS:
    """
    Find the Graf Plane for the hip US module.

    :param hip_datas: The hip data.
    :param results: The results of the segmentation.

    :return: The hip data with the Graf Plane.
    """

    if config.hip.graf_algo_threshold:
        GRAF_THRESHOLD = config.hip.graf_algo_threshold
    else:
        # NOTE(adamcarthur) - this is so that previous behavior is maintained
        GRAF_THRESHOLD = 1

    any_good_graf_data = [
        (hip_data, seg_frame_objs)
        for hip_data, seg_frame_objs in zip(hip_datas, results)
        if hip_data.metrics and all(metric.value != 0 for metric in hip_data.metrics)
    ]

    if len(any_good_graf_data) == 0:
        ulogger.warning("No good graf frames found")
        hip_datas.recorded_error.append("No Perfect Grafs Frames found.")
        hip_datas.recorded_error.critical = True
        any_good_graf_frames = hip_datas.hip_datas
        any_good_graf_data = zip(any_good_graf_frames, results)
    else:
        any_good_graf_frames, _ = zip(*any_good_graf_data)

    # get max alpha frame
    max_alpha = max(
        any_good_graf_frames,
        key=lambda hip_data: hip_data.get_metric(MetricUS.ALPHA),
    ).get_metric(MetricUS.ALPHA)

    all_illiums = [
        hip_data for hip_data in hip_datas if hip_data.get_metric(MetricUS.ALPHA) != 0
    ]

    if len(all_illiums) > 0:
        hip_datas.graf_confs = [
            graf_frame_algo(
                (hip_data, seg_frame_objs),
                max_alpha,
                all_illiums[0].frame_no,
                all_illiums[-1].frame_no,
                None,
                config,
            )
            / GRAF_THRESHOLD
            for hip_data, seg_frame_objs in zip(hip_datas, results)
        ]

    if len(all_illiums) != 0:
        first_illium_frame = all_illiums[0].frame_no
        last_illium_frame = all_illiums[-1].frame_no

    if max_alpha == 0:
        hip_datas.recorded_error.append("Max Alpha is 0.")
        hip_datas.recorded_error.critical = True
        return hip_datas

    if not hasattr(hip_datas, "file_id"):
        hip_datas.file_id = None

    graf_hip, _ = max(
        any_good_graf_data,
        key=lambda hip_zipped_data: graf_frame_algo(
            hip_zipped_data,
            max_alpha,
            first_illium_frame,
            last_illium_frame,
            hip_datas.file_id,
            config,
        ),
    )

    # pick the index closest to the center
    center = len(hip_datas.hip_datas) // 2
    graf_frame = min(
        [graf_hip.frame_no],
        key=lambda index: abs(index - center),
    )

    best_conf = hip_datas.graf_confs[graf_frame]
    if best_conf < 1:
        ulogger.warning("No high-quality Graf Frames found")
        hip_datas.recorded_error.append("No High-Quality Graf Frames found.")
        hip_datas.recorded_error.critical = True
        return hip_datas

    hip_datas.graf_frame = graf_frame

    hip_datas.grafs_hip = [hip for hip in hip_datas if hip.frame_no == graf_frame][0]

    return hip_datas
