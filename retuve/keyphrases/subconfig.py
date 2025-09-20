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
All the subconfigs for Retuve
"""

import os
import sys
from typing import Any, Dict, List, Literal, Tuple, Union

from PIL import ImageFont
from pyparsing import C

from retuve.keyphrases.enums import (
    ACASplit,
    Colors,
    CRFem,
    Curvature,
    GrafSelectionMethod,
    HipMode,
    MetricUS,
    MidLineMove,
    Outputs,
)
from retuve.typehints import GeneralModeFuncType
from retuve.utils import RETUVE_DIR


class HipConfig:
    def __init__(
        self,
        midline_color: Colors,
        aca_split: ACASplit,
        cr_fem_pos: CRFem,
        fem_extention: float,
        midline_move_method: MidLineMove,
        curvature_method: Curvature,
        measurements: List[MetricUS],
        z_gap: float,
        display_frame_no: bool,
        display_side: bool,
        draw_midline: bool,
        display_fem_guess: bool,
        draw_side_metainfo: bool,
        allow_flipping: bool,
        display_bad_frame_reasons: bool,
        graf_selection_method: GrafSelectionMethod,
        graf_selection_func: Any,
        graf_selection_func_args: Dict[str, Any],
        display_graf_conf: bool,
        graf_algo_threshold: float,
        graf_frame_selection: int,
        allow_irregular_illiums: bool,
        allow_horizontal_flipping: bool,
        full_metric_functions: List[Tuple[str, Any]] = None,
        seg_preprocess_functions: List[Any] = None,
        per_frame_metric_functions: List[Tuple[str, Any]] = None,
        post_draw_functions: List[Any] = None,
        use_polyfit_replace_apex: bool = True,
        count_0_coverage_as_error: bool = False,
    ):
        """
        The Hip Subconfig.

        :param midline_color (Colors): The color of the midline.
        :param aca_split (ACASplit): The ACA split method.
        :param cr_fem_pos (CRFem): The CR Femoral position.
        :param fem_extention (float): The femoral extention length.
        :param midline_move_method (MidLineMove): The midline move method.
        :param curvature_method (Curvature): The curvature method.
        :param measurements (list): The measurements to calculate.
        :param z_gap (float): The z gap.
        :param display_frame_no (bool): Display the frame number.
        :param display_side (bool): Display the side.
        :param draw_midline (bool): Draw the midline.
        :param display_fem_guess (bool): Display the femoral guess.
        :param draw_side_metainfo (bool): Draw the side metainfo.
        :param allow_flipping (bool): Allow use of the 3D US Orientaition Algorithm.
        :param display_bad_frame_reasons (bool): Display the bad frame reasons.
        :param graf_selection_method (GrafSelectionMethod): The graf selection method.
        :param graf_selection_func (Any): The graf selection func, if needed.
        :param graf_selection_func_args (dict): The arguments to pass to the graf, if needed.
        :param display_graf_conf (bool): Display the graf confidence.
        :param graf_algo_threshold (float): The graf algorithm confidence threshold.
        :param graf_frame_selection (int): The frame to select for the GRAF MANUAL FRAME algorithm.
        :param allow_irregular_illiums (bool): Allow irregular illiums.
        :param full_metric_functions (list[tuple[str, callable]]): Additional full metrics to compute. Each tuple is (name, func).
            The callable may return one of the following:
            - numeric value (e.g., float or int)
            - (numeric value, dict_of_dev_metrics)
            - dict_of_dev_metrics (if no numeric value is needed)
        :param seg_preprocess_functions (list[callable|tuple[str, callable]]): Functions to run on segmentation results after pre_process_segs_us. Accept either callables or (name, callable) tuples. Each receives (results, config) and may return modified results.
        :param per_frame_metric_functions (list[tuple[str, callable]]): Functions that compute custom per-frame metrics. Each tuple is (name, func). The callable MUST have signature (hip, seg_frame_objs, config) and MUST return (value, dev_metrics_dict). If value is None, it is ignored. dev_metrics_dict may be empty.
        :param post_draw_functions (list[callable|tuple[str, callable]]): Drawing hooks executed after base drawing. Each callable MUST have signature (hip, overlay, config) and MUST return an Overlay (or None to skip). For US, use hip.side to detect Graf/Ant/Post as needed.
        :param use_polyfit_replace_apex (bool): Use polyfit to replace apex outliers.
        :param count_0_coverage_as_error (bool): Count 0 coverage as an error.
        """
        self.midline_color = midline_color
        self.aca_split = aca_split
        self.cr_fem_pos = cr_fem_pos
        self.fem_extention = fem_extention
        self.midline_move_method = midline_move_method
        self.curvature_method = curvature_method
        self.measurements = measurements
        self.z_gap = z_gap
        self.display_frame_no = display_frame_no
        self.display_side = display_side
        self.draw_midline = draw_midline
        self.display_fem_guess = display_fem_guess
        self.draw_side_metainfo = draw_side_metainfo
        self.allow_flipping = allow_flipping
        self.display_bad_frame_reasons = display_bad_frame_reasons
        self.graf_selection_method = graf_selection_method
        self.graf_selection_func = graf_selection_func
        self.graf_selection_func_args = graf_selection_func_args
        self.display_graf_conf = display_graf_conf
        self.graf_algo_threshold = graf_algo_threshold
        self.graf_frame_selection = graf_frame_selection
        self.allow_irregular_illiums = allow_irregular_illiums
        self.allow_horizontal_flipping = allow_horizontal_flipping
        self.full_metric_functions = full_metric_functions or []
        self.seg_preprocess_functions = seg_preprocess_functions or []
        self.per_frame_metric_functions = per_frame_metric_functions or []
        self.post_draw_functions = post_draw_functions or []
        self.use_polyfit_replace_apex = use_polyfit_replace_apex
        self.count_0_coverage_as_error = count_0_coverage_as_error


class TrakConfig:
    def __init__(
        self,
        datasets: List[str],
    ):
        """
        Initialize TrakConfig.

        :param datasets (list): The datasets to process.
        """
        self.datasets = datasets


class BatchConfig:
    def __init__(
        self,
        mode_func: GeneralModeFuncType,
        mode_func_args: Dict[str, Any],
        hip_mode: HipMode,
        processes: int,
        input_types: List[Literal[".dcm", ".jpg", ".png"]],
        datasets: List[str],
    ):
        """
        Initialize BatchConfig.

        :param mode_func (callable): The mode function to use.
        :param mode_func_args (dict): The arguments to pass to the mode function.
        :param hip_mode (str): The mode of the hip us.
        :param processes (int): The number of processes to use.
        :param input_types (list): The type of input to use.
        :param datasets (list): The datasets to process.

        :attr outputs (list): The outputs to generate.
        """
        self.mode_func = mode_func
        self.mode_func_args = mode_func_args
        self.hip_mode = hip_mode
        self.processes = processes
        self.input_types = input_types
        self.datasets = datasets

        # set properly upon registration
        self.outputs: List[Outputs] = [Outputs.METRICS]

    def register(self):
        if self.hip_mode == HipMode.US3D:
            self.outputs.append(Outputs.VIDEO_CLIP)
            self.outputs.append(Outputs.VISUAL3D)

        elif self.hip_mode == HipMode.US2D:
            self.outputs.append(Outputs.IMAGE)

        elif self.hip_mode == HipMode.US2DSW:
            self.outputs.append(Outputs.IMAGE)
            self.outputs.append(Outputs.VIDEO_CLIP)

        elif self.hip_mode == HipMode.XRAY:
            self.outputs.append(Outputs.IMAGE)


class APIConfig:
    def __init__(
        self,
        savedir: str,
        url: str,
        db_path: str,
        upload_dir: str,
        hippa_logging_file: str,
        api_token: str,
        origins: List[str],
        zero_trust: bool,
        zero_trust_interval: int,
        orthanc_url: str,
        orthanc_username: str,
        orthanc_password: str,
    ):
        """
        Initialize APIConfig.

        :param savedir (str): The save directory of results.
        :param url (str): The url to serve the API.
        :param db_path (str): The db path.
        :param upload_dir (str): The upload directory.
        :param hippa_logging_file (str): The logging file for HIPPA related logs.
        :param api_token (str): The api token.
        :param origins (list): The origins to allow.
        :param zero_trust (bool): Zero trust mode.
        :param zero_trust_interval (int): The zero trust interval.
        :param orthanc_url (str): The orthanc url.
        :param orthanc_username (str): The orthanc username.
        :param orthanc_password (str): The orthanc password.
        """
        self.savedir = savedir
        self.url = url
        self.db_path = db_path
        self.upload_dir = upload_dir
        self.hippa_logging_file = hippa_logging_file
        self.api_token = api_token
        self.origins = origins
        self.zero_trust = zero_trust
        self.zero_trust_interval = zero_trust_interval
        self.orthanc_url = orthanc_url
        self.orthanc_username = orthanc_username
        self.orthanc_password = orthanc_password


class VisualsConfig:
    def __init__(
        self,
        font_h1: Union[None, ImageFont.ImageFont],
        font_h2: Union[None, ImageFont.ImageFont],
        default_font_size: int,
        seg_color: Colors,
        seg_alpha: Colors,
        hip_color: Colors,
        bounding_box_color: Colors,
        bounding_box_thickness: int,
        text_color: Colors,
        background_text_color: Colors,
        points_color: Colors,
        points_radius: int,
        line_color: Colors,
        line_thickness: int,
        graf_color: Colors,
        display_full_metric_names: bool,
        display_boxes: bool,
        display_segs: bool,
        min_vid_fps: int,
        min_vid_length: int,
    ):
        """
        The Visuals Subconfig.

        :param font_h1 (ImageFont): The font for the h1.
        :param font_h2 (ImageFont): The font for the h2.
        :param default_font_size (int): The default font size.
        :param seg_color (Colors): The color of segmentations.
        :param seg_alpha (Colors): The alpha of segmentations.
        :param hip_color (Colors): The color of hips.
        :param bounding_box_color (Colors): The color of bounding boxes.
        :param bounding_box_thickness (int): The thickness of bounding boxes.
        :param text_color (Colors): The color of text.
        :param background_text_color (Colors): The background color of text.
        :param points_color (Colors): The color of points.
        :param points_radius (int): The radius of points.
        :param line_color (Colors): The color of lines.
        :param line_thickness (int): The thickness of lines.
        :param graf_color (Colors): The color of grafs.
        :param display_full_metric_names (bool): Display full metric names.
        :param display_boxes (bool): Display boxes.
        :param display_segs (bool): Display segmentations.
        :param min_vid_fps (int): The minimum video fps.
        :param min_vid_length (int): The minimum video length
        """
        self.seg_color = seg_color
        self.seg_alpha = seg_alpha
        self.hip_color = hip_color
        self.bounding_box_color = bounding_box_color
        self.bounding_box_thickness = bounding_box_thickness
        self.text_color = text_color
        self.background_text_color = background_text_color
        self.points_color = points_color
        self.points_radius = points_radius
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.graf_color = graf_color
        self.display_full_metric_names = display_full_metric_names
        self.display_boxes = display_boxes
        self.display_segs = display_segs
        self.min_vid_fps = min_vid_fps
        self.min_vid_length = min_vid_length

        self.font_h1 = font_h1
        self.font_h2 = font_h2
        self.default_font_size = default_font_size

        if not font_h1:
            self.font_h1 = ImageFont.truetype(
                f"{RETUVE_DIR}/files/RobotoMono-Regular.ttf",
                self.default_font_size,
            )

        if not font_h2:
            self.font_h2 = ImageFont.truetype(
                f"{RETUVE_DIR}/files/RobotoMono-Regular.ttf",
                self.default_font_size,
            )
