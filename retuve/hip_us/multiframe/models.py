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
All the code for building the different 3D Models and Visualizations is in this file.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from numpy.typing import NDArray
from scipy.spatial import Delaunay, _qhull

from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.hip_us.classes.enums import HipLabelsUS, Side
from retuve.hip_us.classes.general import HipDatasUS
from retuve.hip_us.metrics.aca import ACA_COLORS, SideZ, Triangle
from retuve.hip_us.typing import CoordinatesArray3D, FemoralHeadSphere
from retuve.keyphrases.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ms(
    x: float, y: float, z: float, radius: float, resolution: int = 20
) -> FemoralHeadSphere:
    """
    Return the coordinates for plotting a sphere centered at (x,y,z)

    :param x: The x-coordinate of the center of the sphere.
    :param y: The y-coordinate of the center of the sphere.
    :param z: The z-coordinate of the center of the sphere.
    :param radius: The radius of the sphere.
    :param resolution: The resolution of the sphere.

    :return: The coordinates for plotting a sphere.
    """
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def build_3d_visual(
    illium_mesh: o3d.geometry.TriangleMesh,
    femoral_sphere: FemoralHeadSphere,
    avg_normals_data: CoordinatesArray3D,
    normals_data: List[Triangle],
    cr_points: List[CoordinatesArray3D],
    hip_datas: HipDatasUS,
    config: Config,
):
    """
    Build the 3D visual for the hip US module.

    :param illium_mesh: The illium mesh.
    :param femoral_sphere: The femoral sphere.
    :param avg_normals_data: The average normals data of the ACA Vectors.
    :param normals_data: The normals data of the ACA Vectors.
    :param cr_points: The CR points.
    :param hip_datas: The hip data.
    :param config: The configuration object.

    :return: The 3D visual.
    """

    import plotly.graph_objects as go

    # Get vertices and faces
    vertices = np.asarray(illium_mesh.vertices)
    faces = np.asarray(illium_mesh.triangles)

    com = np.mean(vertices, axis=0)

    z_gap = config.hip.z_gap * (200 / len(hip_datas))

    # Move the points to the center of mass
    vertices = vertices - com

    # Create figure
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="grey",
                opacity=0.95,
            )
        ]
    )

    if femoral_sphere is not None:
        fig.add_trace(
            go.Surface(
                x=femoral_sphere[0] - com[0],
                y=femoral_sphere[1] - com[1],
                z=femoral_sphere[2] - com[2],
                opacity=0.2,
                colorscale=[[0, "blue"], [1, "blue"]],
                showscale=False,
            )
        )

    if cr_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0] - com[0] for point in cr_points],
                y=[point[1] - com[1] for point in cr_points],
                z=[point[2] - com[2] for point in cr_points],
                mode="markers",
                marker=dict(color="black", size=7),
                showlegend=False,
            )
        )

    # Red, Orange, Yellow, Blue, Cyan, Green
    # [0, 2, 3, 2, 3, 3]
    # https://github.com/plotly/plotly.js/issues/3613#issuecomment-1750709712
    scaling_factors = {
        "red": 0,
        "orange": 2,
        "yellow": 3,
        "blue": 2,
        "cyan": 3,
        "lightgreen": 3,
    }

    for i, color in enumerate(ACA_COLORS.values()):
        triangles = [triangle for triangle in normals_data if triangle.color == color]
        fig.add_trace(
            go.Cone(
                x=[triangle.centroid[0] - com[0] for triangle in triangles],
                y=[triangle.centroid[1] - com[1] for triangle in triangles],
                z=[triangle.centroid[2] - com[2] for triangle in triangles],
                u=[triangle.normal[0] for triangle in triangles],
                v=[triangle.normal[1] for triangle in triangles],
                w=[triangle.normal[2] for triangle in triangles],
                colorscale=[
                    [0, color],
                    [1, color],
                ],
                showscale=False,
                sizemode="absolute",
                sizeref=5 - scaling_factors[color],
                anchor="tail",
            )
        )

    for avg_normal, avg_centroid, color in avg_normals_data:
        fig.add_trace(
            go.Cone(
                x=[avg_centroid[0] - com[0]],
                y=[avg_centroid[1] - com[1]],
                z=[avg_centroid[2] - com[2]],
                u=[avg_normal[0]],
                v=[avg_normal[1]],
                w=[avg_normal[2]],
                colorscale=[[0, color], [1, "black"]],
                showscale=False,
                sizemode="absolute",
                sizeref=32,
                anchor="tail",
            )
        )

    illium_landmarks = []
    fem_landmarks = []
    for hip in hip_datas:
        landmark = hip.landmarks
        if landmark is None:
            continue

        z_pos = z_gap * hip.frame_no
        for point in ["left", "right", "apex"]:
            illium_landmarks.append(list(getattr(landmark, point)) + [z_pos])

        if landmark.point_D is not None:
            fem_landmarks.extend(
                [
                    list(getattr(landmark, attr)) + [z_pos]
                    for attr in ["point_D", "point_d"]
                ]
            )

    for landmarks, color in [
        (illium_landmarks, "black"),
        (fem_landmarks, "blue"),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0] - com[0] for point in landmarks],
                y=[point[1] - com[1] for point in landmarks],
                z=[point[2] - com[2] for point in landmarks],
                mode="markers",
                marker=dict(color=color, size=2),
                showlegend=False,
            )
        )

    # https://plotly.com/python/3d-camera-controls/
    camera = dict(
        up=dict(x=0, y=-1, z=0),  # Pointing in the negative y direction
        eye=dict(
            x=1.5, y=-1.5, z=-1.5
        ),  # Position in the positive x, negative y, negative z region
    )

    fig.update_layout(scene_camera=camera)

    fig.update_layout(scene_dragmode="orbit")

    # add legend for each color
    for apex_side in [SideZ.LEFT, SideZ.RIGHT]:
        for third in Side.ALL():
            fig.add_trace(
                go.Scatter3d(
                    x=[(0, 0, 0)],
                    y=[(0, 0, 0)],
                    z=[(0, 0, 0)],
                    mode="markers",
                    marker=dict(color=ACA_COLORS[apex_side, third]),
                    name=f"{apex_side} {Side.get_name(third)}",
                )
            )

    return fig


def get_femoral_sphere(
    hip_datas: HipDatasUS, config: Config
) -> Optional[Tuple[FemoralHeadSphere, float]]:
    """
    Get the femoral sphere.

    :param hip_datas: The hip data.
    :param config: The configuration object.

    :return: The femoral sphere.
    """
    diameters = []

    z_gap = config.hip.z_gap * (200 / len(hip_datas))

    hips_of_interest = [
        hip for hip in hip_datas if hip.landmarks and hip.landmarks.point_d is not None
    ]

    if len(hips_of_interest) != 0:
        middle_percent = 0.95
        # Calculate how many elements to skip from both ends
        skip_count = int((1 - middle_percent) / 2 * len(diameters))

        # Extract the middle 'middle_percent' percentage of elements
        hips_of_interest = hips_of_interest[
            skip_count : -skip_count if skip_count != 0 else None
        ]

        middle_frame = (
            hips_of_interest[0].frame_no + hips_of_interest[-1].frame_no
        ) // 2
        middle_hip = min(
            hips_of_interest,
            key=lambda hip: abs(hip.frame_no - middle_frame),
        )

        diameter = np.linalg.norm(
            np.array(middle_hip.landmarks.point_D)
            - np.array(middle_hip.landmarks.point_d)
        )

        fem_center = (
            middle_hip.landmarks.point_D[0]
            + (middle_hip.landmarks.point_d[0] - middle_hip.landmarks.point_D[0]) / 2,
            middle_hip.landmarks.point_D[1]
            + (middle_hip.landmarks.point_d[1] - middle_hip.landmarks.point_D[1]) / 2,
            middle_hip.frame_no * z_gap,
        )

        radius = (diameter / 2) + (diameter * 0.05)

        femoral_head_sphere = ms(
            fem_center[0],
            fem_center[1],
            fem_center[2],
            radius,
            resolution=20,
        )

        return femoral_head_sphere, radius

    return None, None


def get_illium_mesh(
    hip_datas: HipDatasUS, results: List[SegFrameObjects], config: Config
) -> Optional[Tuple[o3d.geometry.TriangleMesh, NDArray[np.float64]]]:
    """
    Get the illium mesh.

    :param hip_datas: The hip data.
    :param results: The segmentation results.
    :param config: The configuration object.

    :return: The illium mesh and the apex points.
    """
    illium_pc = []
    apex_points = []

    even_count = 0

    z_gap = config.hip.z_gap * (200 / len(hip_datas))

    for hip_data, seg_frame_objs in zip(hip_datas, results):
        illium = [
            seg_obj
            for seg_obj in seg_frame_objs
            if seg_obj.cls == HipLabelsUS.IlliumAndAcetabulum
        ]

        if (
            len(illium) != 0
            and illium[0].midline is not None
            and hip_data.landmarks
            and hip_data.landmarks.apex is not None
        ):
            illium: SegObject = illium[0]

            # Convert white_points to a convenient shape for vector operations
            midline_moved = np.array(illium.midline_moved)[
                :, ::-1
            ]  # Reverse each point only once

            if even_count % 2 == 0:
                # Pick points a 10 pixel intervals on x axis
                chosen_indexs = np.arange(0, midline_moved.shape[0], 10)
            else:
                # choose every 10 on a 5 point offset
                chosen_indexs = np.arange(5, midline_moved.shape[0], 10)

            for points in midline_moved[chosen_indexs]:
                points = np.append(points, [hip_data.frame_no * z_gap], axis=0)
                illium_pc.append(points)
                apex = hip_data.landmarks.apex

                apex_points.append([apex[0], apex[1], hip_data.frame_no * z_gap])

        even_count += 1

    if len(illium_pc) == 0:
        return None, None

    illium_pc = np.array(illium_pc)

    # use these points to create a surface
    # Create Open3D mesh
    illium_pc_o3d = o3d.geometry.PointCloud()
    illium_pc_o3d.points = o3d.utility.Vector3dVector(illium_pc)

    illium_points_xz = illium_pc[:, [0, 2]]  # Selecting X and Z coordinates

    try:
        # Apply Delaunay triangulation on the XZ plane
        tri = Delaunay(illium_points_xz)
    except _qhull.QhullError:
        return None, None

    # Reconstruct 3D triangles using original Y-coordinates
    vertices = illium_pc  # Using original 3D points as vertices
    triangles = tri.simplices

    # convert to trimesh
    illium_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d.utility.Vector3dVector(vertices)),
        faces=np.asarray(o3d.utility.Vector3iVector(triangles)),
    )

    # apply humphrey smoothing
    illium_mesh = trimesh.smoothing.filter_humphrey(
        illium_mesh, iterations=20, alpha=0.1, beta=1
    )

    # convert back to open3d
    illium_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(illium_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(illium_mesh.faces),
    )

    return illium_mesh, apex_points


def circle_radius_at_z(sphere_radius: float, z_center: float, z_input: float) -> float:
    """
    Calculate the radius of a circle on a sphere at a given Z-coordinate.

    :param sphere_radius: The radius of the sphere.
    :param z_center: The Z-coordinate of the center of the sphere.
    :param z_input: The Z-coordinate of the circle.

    :return: The radius of the circle.
    """
    delta_z = abs(z_input - z_center)
    if delta_z > sphere_radius:
        return 0  # The input Z-coordinate is outside the sphere.
    return np.sqrt(sphere_radius**2 - delta_z**2)
