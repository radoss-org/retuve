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
All unique drawing functions for Hip Ultrasound
"""

import io
import textwrap
import time
from typing import List, Tuple

import cv2
import numpy as np
from attr import has
from PIL import Image, ImageOps
from radstract.data.nifti import NIFTI, convert_images_to_nifti_labels

from retuve.classes.draw import Overlay
from retuve.classes.seg import SegFrameObjects
from retuve.draw import (
    TARGET_SIZE,
    draw_landmarks,
    draw_seg,
    resize_data_for_display,
)
from retuve.hip_us.classes.enums import Side
from retuve.hip_us.classes.general import HipDatasUS, HipDataUS
from retuve.hip_us.handlers.side import get_side_metainfo
from retuve.hip_us.metrics.alpha import draw_alpha
from retuve.hip_us.metrics.coverage import draw_coverage
from retuve.hip_us.multiframe import FemSphere
from retuve.hip_us.multiframe.models import circle_radius_at_z
from retuve.keyphrases.config import Config
from retuve.logs import log_timings, ulogger


def draw_fem_head(
    hip: HipDataUS, fem_sph: FemSphere, overlay: Overlay, z_gap: float
) -> Overlay:
    """
    Draw the femoral head on the image

    :param hip: The Hip Datas
    :param fem_sph: The Femoral Sphere
    :param overlay: The Overlay to draw on

    :return: The Drawn Overlay
    """
    # Get radius at z
    radius = circle_radius_at_z(fem_sph.radius, fem_sph.center[2], z_gap * hip.frame_no)

    # draw the circle
    overlay.draw_circle((fem_sph.center[0], fem_sph.center[1]), radius)

    return overlay


def draw_hips_us(
    hip_datas: HipDatasUS,
    results: List[SegFrameObjects],
    fem_sph: FemSphere,
    config: Config,
) -> tuple[List[np.ndarray], NIFTI]:
    """
    Draw the hip ultrasound images

    :param hip_datas: The Hip Datas
    :param results: The Segmentation Results
    :param fem_sph: The Femoral Sphere
    :param config: The Config

    :return: The Drawn Images
    """
    draw_timings = []
    image_arrays = []
    nifti_frames = []
    nifti = None

    for i, (hip, seg_frame_objs) in enumerate(zip(hip_datas, results)):
        start = time.time()

        final_hip, final_seg_frame_objs, final_image = resize_data_for_display(
            hip, seg_frame_objs
        )

        overlay = Overlay((final_image.shape[0], final_image.shape[1], 3), config)

        overlay = draw_seg(final_seg_frame_objs, overlay, config)

        overlay = draw_landmarks(final_hip, overlay)

        overlay = draw_alpha(final_hip, overlay, config)
        overlay = draw_coverage(final_hip, overlay, config)

        if fem_sph and config.hip.display_fem_guess:
            overlay = draw_fem_head(
                final_hip,
                fem_sph,
                overlay,
                config.hip.z_gap,
            )

        graf_conf = None
        if hasattr(hip_datas, "graf_confs"):
            graf_conf = hip_datas.graf_confs[hip.frame_no]

        overlay, is_graf = draw_other(
            final_hip,
            final_seg_frame_objs,
            hip_datas.graf_frame,
            overlay,
            final_image.shape[:2],
            config,
            graf_conf,
        )

        # Run any custom post-draw hooks after base drawing (standard signature)
        post_draw_funcs = getattr(config.hip, "post_draw_functions", []) or []
        for name, func in [
            (
                (pdf[0], pdf[1])
                if isinstance(pdf, tuple)
                else (getattr(pdf, "__name__", None), pdf)
            )
            for pdf in post_draw_funcs
        ]:
            try:
                out = func(final_hip, overlay, config)
                if isinstance(out, Overlay):
                    overlay = out
            except Exception as e:
                ulogger.error(
                    f"Post-draw function {getattr(func, '__name__', str(func))} failed: {e}"
                )

        if config.hip.display_bad_frame_reasons and hasattr(
            hip_datas, "bad_frame_reasons"
        ):
            if hip.frame_no in hip_datas.bad_frame_reasons:
                overlay.draw_text(
                    hip_datas.bad_frame_reasons[hip.frame_no],
                    final_image.shape[1] // 2,
                    final_image.shape[0] - 100,
                    header="h2",
                )

        img = overlay.apply_to_image(final_image)

        if config.seg_export:
            original_image = seg_frame_objs.img
            test = overlay.get_nifti_frame(
                seg_frame_objs,
                # NOTE(sharpz7) I do not know why this needs to be reversed
                (original_image.shape[1], original_image.shape[0]),
            )
            nifti_frames.append(test)

        # if its the graf frame, append 5 copies
        repeats = len(results) // 6 if is_graf else 1
        for _ in range(repeats):
            image_arrays.append(img)

        draw_timings.append(time.time() - start)

    start = time.time()
    if config.seg_export:
        frames = nifti_frames

        if "Swapped Post and Ant" in hip_datas.recorded_error.errors:
            frames = nifti_frames[::-1]

        # convert to nifti
        nifti = convert_images_to_nifti_labels(frames)
    else:
        nifti = None

    if (time.time() - start) > 0.1:
        log_timings([time.time() - start], title="Nifti Conversion Speed:")

    log_timings(draw_timings, title="Drawing Speed:")

    return image_arrays, nifti


def draw_other(
    hip: HipDataUS,
    seg_frame_objs: SegFrameObjects,
    graf_frame: int,
    overlay: Overlay,
    shape: tuple,
    config: Config,
    graf_conf: float = None,
) -> Tuple[Overlay, bool]:
    """
    Draw the other meta information on the image

    :param hip: The Hip Data
    :param seg_frame_objs: The Segmentation Frame Objects
    :param graf_frame: The Graf Frame
    :param overlay: The Overlay
    :param config: The Config
    :param shape: The Shape of the Image

    :return: The Drawn Overlay
    """

    if config.hip.display_frame_no:
        # Draw Frame No
        overlay.draw_text(
            f"Frame: {hip.frame_no}",
            0,
            0,
            header="h1",
        )

    if graf_conf is not None and config.hip.display_graf_conf:
        overlay.draw_text(
            f"Graf Confidence: {graf_conf:.2f}",
            shape[0] - 100,
            0,
            header="h1",
        )

    is_graf = hip.frame_no == graf_frame and hip.landmarks is not None

    # Check if graf alpha
    if is_graf:
        overlay.draw_text(
            f"Grafs Frame",
            int(hip.landmarks.left[0]),
            int(hip.landmarks.left[1] - 120),
            header="h2",
            grafs=True,
        )

        # add a border
        overlay.draw_box(
            (0, 0, shape[1], shape[0]),
            grafs=True,
        )

    if hip.landmarks and hip.landmarks.right and config.hip.display_side:
        xl, yl = hip.landmarks.right
        overlay.draw_text(
            f"side: {Side.get_name(hip.side)}",
            int(xl),
            int(yl),
            header="h2",
        )

    if config.hip.draw_midline:
        for seg_obj in seg_frame_objs:
            if seg_obj.midline_moved is not None:
                overlay.draw_skeleton(seg_obj.midline_moved)

    if config.hip.draw_side_metainfo:
        closest_illium, mid = get_side_metainfo(hip, seg_frame_objs)
        # these are points, draw them
        if closest_illium is not None:
            overlay.draw_cross(closest_illium)
            overlay.draw_cross(mid)

    return overlay, is_graf


def draw_table(shape: tuple, hip_datas: HipDatasUS) -> np.ndarray:
    """
    Draw the table of the metrics onto an image

    :param shape: The Shape of the Image
    :param hip_datas: The Hip Datas

    :return: The Image with the Table and any errors
    """
    import plotly.graph_objects as go

    start = time.time()

    # Create empty image with the specified shape
    empty_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)

    # Find new shape by running 1024 algo
    shape = ImageOps.contain(Image.fromarray(empty_img), (TARGET_SIZE)).size[:2]

    headers = [""] + hip_datas.metrics[0].names()
    values = []

    for metrics in reversed(hip_datas.metrics):
        values.append(metrics.dump())

    # Rotate and transpose the values list
    values = list(zip(*values[::-1]))

    # Create a Plotly figure for the table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, font=dict(size=18), height=50),
                cells=dict(
                    values=values,
                    fill=dict(color=["paleturquoise", "white"]),
                    font=dict(size=18),
                    height=50,
                ),
            )
        ]
    )

    MARGIN = 30
    fig.update_layout(
        autosize=False,
        width=shape[1],
        height=shape[0],
        margin=dict(l=MARGIN, r=MARGIN, b=MARGIN, t=MARGIN),
    )

    # Save the figure as bytes
    img_bytes = fig.to_image(format="png", width=shape[1], height=shape[0])

    # Convert image bytes to numpy array
    data_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    # Text Wrapping Logic
    recorded_error_text = str(hip_datas.recorded_error)

    # Set the maximum width for the text box in pixels
    max_text_width = data_image.shape[1] - 20  # Margin of 10px from both sides
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2

    # Estimate the width of each character based on the font
    (text_width, text_height), _ = cv2.getTextSize(
        recorded_error_text, font, font_scale, font_thickness
    )
    char_width = (
        text_width // len(recorded_error_text) if recorded_error_text else 1
    )  # Estimate width of one char

    # Wrap the text based on character width and max text width
    wrap_width = max_text_width // char_width
    wrapped_text = textwrap.fill(recorded_error_text, width=wrap_width)

    # Split the wrapped text into lines
    lines = wrapped_text.split("\n")

    # Draw each line of wrapped text
    y = data_image.shape[0] - 300  # Starting y position
    line_height = text_height + 10  # Spacing between lines
    for line in lines:
        # Put each line of text on the image
        cv2.putText(
            data_image,
            line,
            (10, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        y += line_height  # Move down for the next line

    # Log the time taken to draw the table
    ulogger.info(f"Table Drawing Time: {time.time() - start:.2f}s")

    return data_image
