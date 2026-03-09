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
This module contains the function to convert landmarks to metrics for hip x-ray.
"""

import time
from typing import List

from retuve.classes.metrics import Metric2D
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.hip_xray.metrics.ace import find_ace
from retuve.hip_xray.metrics.ihdi import find_ihdi
from retuve.hip_xray.metrics.tonnis import find_tonnis
from retuve.hip_xray.metrics.wiberg import find_wiberg
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

# Configuration mapping metric details to their calculation functions.
# This makes adding/removing metrics a single-line change.
METRIC_CONFIG = [
    {"base_name": "ace", "suffix": "index", "func": find_ace},
    {"base_name": "wiberg", "suffix": "index", "func": find_wiberg},
    {"base_name": "ihdi", "suffix": "grade", "func": find_ihdi},
    {"base_name": "tonnis", "suffix": "grade", "func": find_tonnis},
]


def landmarks_2_metrics_xray(
    list_landmarks: List[LandmarksXRay],
    config: Config,
) -> List[HipDataXray]:
    """
    Convert the landmarks to metrics for hip x-ray.

    :param list_landmarks: The List of Landmarks.
    :param config: The Config.
    :return: The List of Hip Data.
    """
    hips = []
    timings = []

    for frame_no, landmarks in enumerate(list_landmarks):
        start = time.time()
        hip = HipDataXray()
        hip.landmarks = landmarks

        # Correctly check if all landmark attributes are None.
        is_empty = all(val is None for val in vars(landmarks).values())

        if is_empty:
            hip.recorded_error.append("No landmarks found.")
            # Populate with zero-value metrics for consistency.
            for metric in METRIC_CONFIG:
                base, suffix = metric["base_name"], metric["suffix"]
                hip.metrics.append(Metric2D(f"{base}_{suffix}_left", 0))
                hip.metrics.append(Metric2D(f"{base}_{suffix}_right", 0))
        else:
            # Calculate and append metrics using the configuration.
            for metric in METRIC_CONFIG:
                base, suffix = metric["base_name"], metric["suffix"]
                value_l, value_r = metric["func"](landmarks)
                value_l, value_r = round(float(value_l), 2), round(float(value_r), 2)
                hip.metrics.append(Metric2D(f"{base}_{suffix}_left", value_l))
                hip.metrics.append(Metric2D(f"{base}_{suffix}_right", value_r))

        hips.append(hip)
        timings.append(time.time() - start)

    log_timings(timings, title="Landmarks->Metrics Speed:")
    return hips
