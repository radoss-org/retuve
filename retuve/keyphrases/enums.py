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
All the Configuration-based Enums
"""

import os

from PIL import ImageColor


class Colors:
    """
    Class for managing common colors.
    """

    WHITE = (255, 255, 255)
    RED = (255, 84, 115)
    GREEN = (0, 255, 0)
    BLUE = (17, 60, 130)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    DARK_RED = (139, 0, 0)
    LIGHT_BLUE = (113, 194, 245)
    PURPLE = (215, 84, 255)
    TEAL = (0, 128, 128)
    GOLD = ImageColor.getcolor("#78620A", "RGB")
    GOLD_LIGHT = (232, 219, 102)
    LIGHT_GREEN = (70, 199, 158)

    def __init__(self, rgb=None):
        self.rgb = rgb

    def rgba(self, alpha=1):
        """
        Get the color with alpha.

        :param alpha: The alpha value
        """
        return (*self.rgb, int(alpha * 255))

    def bgr(self):
        """
        Get the color in BGR format.
        """
        return self.rgb[::-1]

    def __repr__(self) -> str:
        return f"Color({self.rgb})"


class ACASplit:
    """
    Options for ACA split.

    :attr THIRDS: Split into thirds equally
    :attr GRAFS: Split using the grafs plane as the divider

    :attr OPTIONS: List of acceptable options
    """

    THIRDS = "thirds"
    GRAFS = "grafs"

    OPTIONS = [THIRDS, GRAFS]


class CRFem:
    """
    Options for Femoral Head Center position.

    :attr GRAFS: Use the GRAFS method
    :attr CENTER: Use the CENTER method

    :attr OPTIONS: List of acceptable options
    """

    GRAFS = "grafs"
    CENTER = "center"

    OPTIONS = [GRAFS, CENTER]


class MidLineMove:
    """
    Options for midline move method.

    :attr BASIC: Use the most basic method.
    """

    BASIC = "basic"


class Curvature:
    """
    Options for curvature method.

    :attr Radist: Use the radist method.
    """

    RADIST = "radist"


class MetricUS:
    """
    The Ultrasound Metric types.

    :attr ALPHA: The alpha angle
    :attr COVERAGE: The coverage
    :attr CURVATURE: The curvature
    :attr CENTERING_RATIO: The centering ratio
    :attr ACA: The Acetabular Coverage Angle
    """

    ALPHA = "alpha"
    COVERAGE = "coverage"
    CURVATURE = "curvature"
    CENTERING_RATIO = "cen. ratio"
    ACA = "aca"

    @classmethod
    def ALL(cls):
        return [
            cls.ALPHA,
            cls.COVERAGE,
            cls.CURVATURE,
            cls.CENTERING_RATIO,
            cls.ACA,
        ]


class OperationType:
    """
    Operation type of the AI Model being used.

    :attr SEG: The model is a segmentation model
    :attr LANDMARK: The model is a landmark model
    """

    SEG = "seg"
    LANDMARK = "landmark"


class HipMode:
    """
    Hip US modes.

    :attr US3D: 3D Ultrasound
    :attr US2D: 2D Ultrasound
    :attr US2DSW: 2D Sweep Ultrasound
    :attr XRAY: X-Ray
    """

    US3D = "us3d"
    US2D = "us2d"
    US2DSW = "us2dsw"
    XRAY = "xray"


class Outputs:
    """
    Output Extensions.

    """

    IMAGE = "img.jpg"
    METRICS = "metrics.json"
    VIDEO_CLIP = "video.mp4"
    VISUAL3D = "visual_3d.html"
    NIFTI = "seg.nii.gz"


class GrafSelectionMethod:
    """
    Graf Frame Selection Methods.
    """

    MANUAL_FEATURES = "manual_features"
    OBJ_CLASSIFY = "obj_classify"
    MANUAL_FRAME = "manual_frame"
