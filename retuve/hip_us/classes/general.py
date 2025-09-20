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
Contains the Hip Data Classes and the Landmarks Class.
"""

from typing import Dict, List, Tuple, Union

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from plotly.graph_objs import Figure
from radstract.data.nifti import NIFTI
from retuve.classes.general import RecordedError
from retuve.classes.metrics import Metric2D, Metric3D
from retuve.hip_us.classes.dev import DevMetricsUS
from retuve.hip_us.classes.enums import Side
from retuve.keyphrases.config import Config


class LandmarksUS:
    """
    Class to store the landmarks of the Hip Ultrasound.

    These are related to a standard graf image of a Hip.

    :attr left: tuple: Left landmark
    :attr right: tuple: Right landmark
    :attr apex: tuple: Apex landmark
    :attr point_D: tuple: Point D landmark
    :attr point_d: tuple: Point d landmark
    :attr mid_cov_point: tuple: Mid cov point landmark
    """

    def __init__(
        self,
        left: Tuple[int, int] = None,
        right: Tuple[int, int] = None,
        apex: Tuple[int, int] = None,
        point_D: Tuple[int, int] = None,
        point_d: Tuple[int, int] = None,
        mid_cov_point: Tuple[int, int] = None,
    ):
        self.left = left
        self.right = right
        self.apex = apex
        self.point_D = point_D
        self.point_d = point_d
        self.mid_cov_point = mid_cov_point

    def __iter__(self) -> Tuple:
        return iter(
            (
                self.left,
                self.right,
                self.apex,
                self.point_D,
                self.point_d,
                self.mid_cov_point,
            )
        )

    def items(self) -> Dict[str, Tuple]:
        return {
            "left": self.left,
            "right": self.right,
            "apex": self.apex,
            "point_D": self.point_D,
            "point_d": self.point_d,
            "mid_cov_point": self.mid_cov_point,
        }.items()

    def __setitem__(self, key: str, value: Tuple[int, int]):
        # use setattr to avoid infinite recursion
        setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"LandmarksUS({self.left}, {self.right}, {self.apex}, "
            f"{self.point_D}, {self.point_d}, {self.mid_cov_point})"
        )

    def to_dict(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns the landmarks as a dictionary.
        """
        return {
            "left": self.left,
            "right": self.right,
            "apex": self.apex,
            "point_D": self.point_D,
            "point_d": self.point_d,
            "mid_cov_point": self.mid_cov_point,
        }


class HipDataUS:
    """
    Class to store the Hip Data of a single Hip Image.

    :attr landmarks: LandmarksUS: Landmarks of the Hip Image.
    :attr metrics: List[Metric2D]: List of 2D Metrics.
    :attr frame_no: int: Frame number of the Hip Image.
    :attr side: Side: Side of the Hip Image.
    """

    def __init__(
        self,
        landmarks: LandmarksUS = None,
        metrics: List[Metric2D] = None,
        frame_no: int = None,
        side: Side = None,
    ):
        self.landmarks = landmarks
        self.frame_no = frame_no
        self.side = side

        self.metrics = metrics

    def __str__(self) -> str:
        return (
            f"HipDataUS(landmarks={self.landmarks}, metrics={self.metrics}, "
            f"frame_no={self.frame_no}, side={self.side})"
        )

    def marked(self) -> bool:
        """
        Returns whether the Hip Image has any non-zero metrics.
        """
        if self.metrics is None:
            return False

        return any(metric.value != 0 for metric in self.metrics)

    def get_metric(self, name: str) -> float:
        """
        Returns the value of a metric by its name.
        """
        if self.metrics is None:
            return 0

        for metric in self.metrics:
            if metric.name == name:
                return metric.value
        return 0

    def json_dump(self, config: Config, dev_metrics: DevMetricsUS):
        """
        Returns a dictionary of the Hip Data in JSON format.

        Used for saving the Hip Data to a JSON file or returning it from an API call.

        :param config: Config object.
        :param dev_metrics: DevMetricsUS object.

        :return: Dictionary of the Hip Data in JSON format.
        """

        if self.metrics is None:
            return {
                "metrics": [],
                "keyphrase": config.name,
                "dev_metrics": dev_metrics.json_dump(),
            }

        # Serialize Metric2D objects as strings for deterministic comparisons
        serialized_metrics: List[str] = [str(m) for m in self.metrics]
        return {
            "metrics": serialized_metrics,
            "keyphrase": config.name,
            "dev_metrics": dev_metrics.json_dump(),
        }


class HipDatasUS:
    """
    Class to store the Hip Data of multiple Hip Images.

    :attr recorded_error: RecordedError: Recorded Error of the Hip Data.
    :attr hip_datas: List[HipDataUS]: List of Hip Data.
    :attr metrics: List[Metric3D]: List of 3D Metrics.
    :attr graf_frame: int: Frame where the Graf angle is calculated.
    :attr grafs_hip: HipDataUS: Hip Data where the Graf angle is calculated.
    :attr dev_metrics: DevMetricsUS: Development Metrics of the Hip Data.
    :attr video_clip: ImageSequenceClip: Video Clip of the Hip Data.
    :attr visual_3d: Figure: 3D Visualization of the Hip Data.
    :attr nifti: NIFTI: NIFTI object of the Hip Data.
    :attr cr_points: List[float]: List of critical points detected.
    """

    def __init__(
        self,
    ):
        self.recorded_error: RecordedError = RecordedError()

        self.hip_datas: List[HipDataUS] = []

        self.metrics: List[Metric3D] = []

        self.graf_frame: int = None
        self.grafs_hip: HipDataUS = None
        self.dev_metrics: DevMetricsUS = None
        # Temporary collector for custom dev metrics from full_metric_functions
        self.dev_metrics_custom: Dict[str, object] = {}
        self.video_clip: ImageSequenceClip = None
        self.visual_3d: Figure = None
        self.nifti: NIFTI = None
        self.cr_points: List[float] = []

    def __iter__(self) -> HipDataUS:
        return iter(self.hip_datas)

    def __getitem__(self, index) -> HipDataUS:
        return self.hip_datas[index]

    def __setitem__(self, index, value):
        self.hip_datas[index] = value

    def __len__(self):
        return len(self.hip_datas)

    def __str__(self):
        output = ""
        for hip_data in self.hip_datas:
            output += f"{hip_data}\n"

        return output

    def append(self, hip_data: HipDataUS):
        """
        Append a HipDataUS object to the HipDatasUS object.
        """
        self.hip_datas.append(hip_data)

    def sorted_metrics(self) -> List[Metric3D]:
        """
        Returns the metrics sorted by name.
        """
        return sorted(self.metrics, key=lambda metric: metric.name)

    def json_dump(self, config: Config) -> Dict[str, Union[dict, str]]:
        """
        Returns a dictionary of the Hip Datas in JSON format.

        Used for saving the Hip Datas to a JSON file or returning it from an API call.

        :param config: Config object.

        :return: Dictionary of the Hip Datas in JSON format.
        """
        return {
            "metrics": [
                {metric.dump()[0]: metric.dump()[1:]} for metric in self.metrics
            ],
            "graf_frame": self.graf_frame,
            "dev_metrics": (self.dev_metrics.json_dump() if self.dev_metrics else None),
            "recorded_error": str(self.recorded_error),
            "keyphrase": config.name,
        }
