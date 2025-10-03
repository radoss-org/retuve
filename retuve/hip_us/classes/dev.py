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
Developer metrics for Ultrasound Analysis
"""

from typing import Dict, List


class DevMetricsUS:
    def __init__(self):
        """
        Class to store the metrics of the development of the US segmentation

        :attr os_ichium_detected: bool: True if the os_ichium is detected
        :attr no_frames_segmented: int: Number of frames segmented
        :attr no_frames_marked: int: Number of frames marked
        :attr graf_frame: int: Frame where the Graf angle is calculated
        :attr acetabular_mid_frame: int: Frame where the acetabular midline is calculated
        :attr fem_mid_frame: int: Frame where the femoral midline is calculated
        :attr critial_error: bool: True if a critical error is detected
        :attr cr_points: list: List of critical points detected
        :attr total_frames: int: Total number of frames in
        """
        self.os_ichium_detected = False
        self.no_frames_segmented = 0
        self.no_frames_marked = 0
        self.graf_frame = 0
        self.acetabular_mid_frame = 0
        self.fem_mid_frame = 0
        self.critial_error = False
        self.cr_points = []
        self.total_frames = 0
        self.average_ilium_angle_baseline = 0
        # Arbitrary custom/dev metrics from plugins or full_metric_functions
        self.custom: Dict[str, object] = {}

    def __repr__(self) -> str:
        return (
            f"DevMetricsUS(os_ichium_detected={self.os_ichium_detected}, "
            f"no_frames_segmented={self.no_frames_segmented}, "
            f"no_frames_marked={self.no_frames_marked}, "
            f"graf_frame={self.graf_frame}, "
            f"acetabular_mid_frame={self.acetabular_mid_frame}, "
            f"fem_mid_frame={self.fem_mid_frame}, "
            f"critial_error={self.critial_error}, "
            f"cr_points={self.cr_points}, "
            f"total_frames={self.total_frames}, "
            f"average_ilium_angle_baseline={self.average_ilium_angle_baseline})"
        )

    def data(self) -> List:
        """
        Return the data of the class

        :return: List of the data of the class
        """
        return [
            self.os_ichium_detected,
            self.no_frames_segmented,
            self.no_frames_marked,
            self.graf_frame,
            self.acetabular_mid_frame,
            self.fem_mid_frame,
            self.critial_error,
            self.cr_points,
            self.total_frames,
            self.average_ilium_angle_baseline,
            self.custom,
        ]

    def json_dump(self) -> Dict[str, List]:
        """
        Return the data of the class in a dictionary

        Useful for saving the data in a json file, or the return of an API call.

        :return: Dictionary of the data of the class
        """
        return {
            "os_ichium_detected": self.os_ichium_detected,
            "no_frames_segmented": self.no_frames_segmented,
            "no_frames_marked": self.no_frames_marked,
            "graf_frame": self.graf_frame,
            "acetabular_mid_frame": self.acetabular_mid_frame,
            "fem_mid_frame": self.fem_mid_frame,
            "critial_error": self.critial_error,
            "cr_points": self.cr_points,
            "total_frames": self.total_frames,
            "average_ilium_angle_baseline": self.average_ilium_angle_baseline,
            "custom": self.custom,
        }
