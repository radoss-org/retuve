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
A function for getting all the other metrics that are just for development purposes.
"""

from typing import List

import numpy as np

from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.dev import DevMetricsUS
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.hip_us.classes.general import HipDatasUS
from retuve.keyphrases.config import Config


def get_dev_metrics(
    hip_datas: HipDatasUS,
    results: List[SegFrameObjects],
    config: Config,
) -> HipDatasUS:
    """
    Get all the development metrics.

    :param hip_datas: HipDatasUS: The HipDatasUS object.
    :param results: List[SegFrameObjects]: The results of the segmentation.
    :param config: Config: The Config object.

    :return: HipDatasUS: The updated HipDatasUS object with the development metrics.
    """
    dev_metrics = DevMetricsUS()

    dev_metrics.graf_frame = hip_datas.graf_frame

    ace_marked_hips = []
    fem_marked_hips = []
    ilium_angle_baselines = []

    for hip_data, seg_frame_objs in zip(hip_datas, results):
        detected = [seg_obj.cls for seg_obj in seg_frame_objs]
        detected_bool = any(detected)

        if detected_bool:
            dev_metrics.no_frames_segmented += 1

        if hip_data.marked():
            dev_metrics.no_frames_marked += 1

        if HipLabelsUS.OsIchium in detected:
            dev_metrics.os_ichium_detected = True

        if HipLabelsUS.IlliumAndAcetabulum in detected:
            ace_marked_hips.append(hip_data)

        if HipLabelsUS.FemoralHead in detected:
            fem_marked_hips.append(hip_data)

    if len(ace_marked_hips) > 0:
        dev_metrics.acetabular_mid_frame = ace_marked_hips[
            len(ace_marked_hips) // 2
        ].frame_no

    if len(fem_marked_hips) > 0:
        dev_metrics.fem_mid_frame = fem_marked_hips[len(fem_marked_hips) // 2].frame_no

    dev_metrics.critial_error = hip_datas.recorded_error.critical

    dev_metrics.cr_points = (
        [
            round(point[2] / (config.hip.z_gap * (200 / len(hip_datas))), 0)
            for point in hip_datas.cr_points
        ]
        if hip_datas.cr_points
        else []
    )

    if dev_metrics.cr_points:
        dev_metrics.cr_points = [
            dev_metrics.cr_points[2],
            dev_metrics.cr_points[1],
            dev_metrics.cr_points[0],
        ]

    dev_metrics.total_frames = len(hip_datas)

    dev_metrics.average_ilium_angle_baseline = round(
        np.mean(hip_datas.ilium_angle_baselines), 2
    )

    # Merge any custom dev metrics collected earlier (e.g., from full_metric_functions)
    try:
        if hasattr(hip_datas, "dev_metrics_custom") and hip_datas.dev_metrics_custom:
            # Initialize the custom dict if absent
            if not isinstance(dev_metrics.custom, dict):
                dev_metrics.custom = {}
            dev_metrics.custom.update(hip_datas.dev_metrics_custom)
    except Exception:
        pass

    hip_datas.dev_metrics = dev_metrics

    return hip_datas
