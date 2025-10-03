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
Class Structs related to Hip X-Ray
"""

from enum import Enum
from typing import Any, Dict, List, Tuple

from retuve.classes.general import RecordedError
from retuve.classes.metrics import Metric2D


class HipLabelsXray(Enum):
    """
    Segmentation Labels for Hip X-Ray
    """

    AceTriangle = 0

    @classmethod
    def get_name(cls, label: "HipLabelsXray"):
        """
        Get the name of the label from the Enum

        :param label: The label
        :return: The name of the label

        """
        if label == HipLabelsXray.AceTriangle:
            return "Acentabular Triangle"
        else:
            return "Unknown"


class DevMetricsXRay:
    """
    Developer Metrics for Hip X-Ray

    Currently, this is a placeholder class.
    """

    def __init__(self):
        pass

    def docs(self):
        return {}

    def __repr__(self) -> str:
        return "DevMetricsXRay()"

    def data(self):
        return []

    def json_dump(self):
        return {}


class LandmarksXRay:
    """
    Landmarks for Hip X-Ray

    :attr pel_l_o: The Outer Pelvis Landmark on the Left
    :attr pel_l_i: The Inner Pelvis Landmark on the Left
    :attr pel_r_o: The Outer Pelvis Landmark on the Right
    :attr pel_r_i: The Inner Pelvis Landmark on the Right
    :attr fem_l: The Femoral Landmark on the Left
    :attr fem_r: The Femoral Landmark on the Right
    """

    def __init__(
        self,
        pel_l_o: Tuple[float, float] = None,
        pel_l_i: Tuple[float, float] = None,
        pel_r_o: Tuple[float, float] = None,
        pel_r_i: Tuple[float, float] = None,
        fem_l: Tuple[float, float] = None,
        fem_r: Tuple[float, float] = None,
        h_point_l: Tuple[float, float] = None,
        h_point_r: Tuple[float, float] = None,
    ):
        self.pel_l_o = pel_l_o
        self.pel_l_i = pel_l_i
        self.pel_r_o = pel_r_o
        self.pel_r_i = pel_r_i

        self.fem_l = fem_l
        self.fem_r = fem_r

        self.h_point_l = h_point_l
        self.h_point_r = h_point_r

    def __str__(self) -> str:
        return (
            f"Landmarks(pel_l_o={self.pel_l_o}, pel_l_i={self.pel_l_i}, "
            f"pel_r_o={self.pel_r_o}, pel_r_i={self.pel_r_i}, "
            f"fem_l={self.fem_l}, fem_r={self.fem_r})"
        )

    def __iter__(self) -> List[Tuple[float, float]]:
        return iter(
            [
                self.pel_l_o,
                self.pel_l_i,
                self.pel_r_o,
                self.pel_r_i,
                self.fem_l,
                self.fem_r,
                self.h_point_l,
                self.h_point_r,
            ]
        )

    def items(self) -> Dict[str, Tuple]:
        return {
            "pel_l_o": self.pel_l_o,
            "pel_l_i": self.pel_l_i,
            "pel_r_o": self.pel_r_o,
            "pel_r_i": self.pel_r_i,
            "fem_l": self.fem_l,
            "fem_r": self.fem_r,
            "h_point_l": self.h_point_l,
            "h_point_r": self.h_point_r,
        }.items()

    def __setitem__(self, key: str, value: Tuple[int, int]):
        # use setattr to avoid infinite recursion
        setattr(self, key, value)


class HipDataXray:
    """
    Hip Data for X-Ray

    :attr metrics: The Metrics
    :attr dev_metrics: The Developer Metrics
    :attr landmarks: The Landmarks
    :attr frame_no: The Frame Number
    :attr recorded_error: The Recorded Error
    """

    def __init__(self):
        self.metrics: List[Metric2D] = []
        self.dev_metrics: DevMetricsXRay = []
        self.landmarks: LandmarksXRay = None
        self.frame_no: int = None
        self.recorded_error: RecordedError = RecordedError()

    def get_metric(self, name: str):
        """
        Returns the value of a metric by its name, or 0 if not found.
        """
        if not self.metrics:
            return 0
        for metric in self.metrics:
            try:
                if metric and metric.name == name:
                    return metric.value
            except Exception:
                continue
        return 0

    def json_dump(self, config, dev_metrics: DevMetricsXRay) -> Dict[str, Any]:
        return {
            "metrics": [{metric.name: metric.value} for metric in self.metrics],
            "keyphrase": config.name,
            "dev_metrics": dev_metrics.json_dump(),
        }
