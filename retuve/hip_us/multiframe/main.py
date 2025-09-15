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

import time
from typing import List

import numpy as np
import open3d as o3d
from numpy import ndarray as NDArray
from plotly.graph_objects import Figure

from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.enums import Side
from retuve.hip_us.classes.general import HipDatasUS, Metric3D
from retuve.hip_us.metrics.aca import get_aca
from retuve.hip_us.metrics.c_ratio import get_centering_ratio
from retuve.hip_us.multiframe.models import (
    build_3d_visual,
    get_femoral_sphere,
    get_illium_mesh,
)
from retuve.hip_us.typing import CoordinatesArray3D, FemoralHeadSphere
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import MetricUS
from retuve.logs import ulogger
from retuve.utils import rmean


class FemSphere:
    def __init__(self, center: CoordinatesArray3D, radius: float):
        self.center = center
        self.radius = radius


def get_3d_metrics_and_visuals(
    hip_datas: HipDatasUS, results: List[SegFrameObjects], config: Config
) -> tuple[
    HipDatasUS,
    Figure,
    FemSphere,
    o3d.geometry.TriangleMesh,
    NDArray[np.float64],
    FemoralHeadSphere,
    CoordinatesArray3D,
    CoordinatesArray3D,
]:
    """
    Get the 3D metrics and visuals for the hip US module.

    :param hip_datas: The hip data.
    :param results: The segmentation results.
    :param config: The configuration object.

    :return: The hip data, the 3D visual, the femoral sphere,
             the illium mesh, the apex points, the femoral sphere,
             the average normals data of the ACA Vectors, and the
             normals data of the ACA Vectors.
    """
    start = time.time()
    illium_mesh, apex_points = get_illium_mesh(hip_datas, results, config)
    visual_3d = None

    if illium_mesh is None:
        hip_datas.recorded_error.append("No Illium Mesh.")
        hip_datas.recorded_error.critical = True

    femoral_sphere, radius = get_femoral_sphere(hip_datas, config)

    if illium_mesh:
        aca, avg_normals_data, normals_data, aca_error_msg = get_aca(
            illium_mesh,
            apex_points,
            hip_datas.grafs_hip,
            len(hip_datas),
            config,
        )

        hip_datas.recorded_error.append(aca_error_msg)
    else:
        aca = None
        avg_normals_data = None
        normals_data = None

    if femoral_sphere and illium_mesh:
        c_ratio, cr_points = get_centering_ratio(
            illium_mesh, femoral_sphere, hip_datas, config
        )
        hip_datas.cr_points = cr_points

    else:
        c_ratio = 0
        cr_points = []

    if illium_mesh and femoral_sphere:
        visual_3d = build_3d_visual(
            illium_mesh,
            femoral_sphere,
            avg_normals_data,
            normals_data,
            cr_points,
            hip_datas,
            config,
        )

    if aca and MetricUS.ACA in config.hip.measurements:
        aca_metric = Metric3D(
            name=MetricUS.ACA,
            graf=aca[Side.GRAF],
            post=aca[Side.POST],
            ant=aca[Side.ANT],
        )
        hip_datas.metrics.append(aca_metric)

    if c_ratio and MetricUS.CENTERING_RATIO in config.hip.measurements:
        c_ratio = Metric3D(
            name=MetricUS.CENTERING_RATIO,
            full=c_ratio,
        )
        hip_datas.metrics.append(c_ratio)

    for name in config.hip.measurements:
        if not any(metric.name == name for metric in hip_datas.metrics):
            post_values = [
                hip_data.get_metric(name)
                for hip_data in hip_datas
                if hip_data.side == Side.POST and hip_data.get_metric(name) != 0
            ] or [0]
            ant_values = [
                hip_data.get_metric(name)
                for hip_data in hip_datas
                if hip_data.side == Side.ANT and hip_data.get_metric(name) != 0
            ] or [0]

            graf_value = 0
            if hip_datas.graf_frame:
                graf_value = hip_datas.grafs_hip.get_metric(name)

            hip_datas.metrics.append(
                Metric3D(
                    name=name,
                    graf=graf_value,
                    post=rmean(post_values),
                    ant=rmean(ant_values),
                )
            )

    if all(metric.post == 0 for metric in hip_datas.metrics):
        hip_datas.recorded_error.append("No Posterior values recorded.")
        hip_datas.recorded_error.critical = True

    if all(metric.ant == 0 for metric in hip_datas.metrics):
        hip_datas.recorded_error.append("No Anterior values recorded.")
        hip_datas.recorded_error.critical = True

    if len(cr_points) != 0:
        fem_sph = FemSphere(cr_points[0], radius)
    else:
        fem_sph = None

    ulogger.info(f"Time for all 3D Elements: {round(time.time() - start, 2)} s")

    return (
        hip_datas,
        visual_3d,
        fem_sph,
        illium_mesh,
        apex_points,
        femoral_sphere,
        avg_normals_data,
        normals_data,
    )
