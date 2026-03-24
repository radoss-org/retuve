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
Metric: Centering Ratio
"""

from typing import Tuple

import numpy as np
import open3d as o3d

from retuve.hip_us.classes.general import HipDatasUS
from retuve.hip_us.typing import CoordinatesArray3D, FemoralHeadSphere
from retuve.keyphrases.config import Config
from retuve.utils import warning_decorator


@warning_decorator(alpha=True)
def get_centering_ratio(
    illium_mesh: o3d.geometry.TriangleMesh,
    femoral_head_sphere: FemoralHeadSphere,
    hip_datas: HipDatasUS = None,
    config: Config = None,
) -> Tuple[float, Tuple[CoordinatesArray3D, CoordinatesArray3D, CoordinatesArray3D]]:
    """
    Get the centering ratio for the femoral head.

    :param illium_mesh: o3d.geometry.TriangleMesh: The Illium mesh.
    :param femoral_head_sphere: FemoralHeadSphere: The femoral head sphere.
    :param hip_datas: HipDatasUS: The HipDatasUS object.
    :param config: Config: The Config object.

    :return: The centering ratio and the 3 points used to calculate it.
    """

    if hip_datas.graf_frame is None:
        return 0, (None, None, None)

    verticies = np.array(illium_mesh.vertices)

    # Get min and maz verticies in the z axis
    # Assume min will be anterior and max will be posterior
    min_z = np.min(verticies[:, 2])
    max_z = np.max(verticies[:, 2])

    fem_center = [
        np.mean(femoral_head_sphere[0]),
        np.mean(femoral_head_sphere[1]),
        np.mean(femoral_head_sphere[2]),
    ]

    min_z_vert = [
        fem_center[0],
        fem_center[1],
        min_z,
    ]
    max_z_vert = [
        fem_center[0],
        fem_center[1],
        max_z,
    ]

    # https://files.mcaq.me/a6jj5.png
    ratio = (fem_center[2] - hip_datas.graf_frame) / (max_z - min_z)

    return round(ratio, 2), (fem_center, max_z_vert, min_z_vert)
