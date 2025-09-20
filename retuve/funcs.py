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
Contains the high-level functions that are used to run the Retuve pipeline.
"""

import copy
import time
from typing import Any, BinaryIO, Callable, Dict, List, Tuple, Union

import pydicom
from moviepy import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
from plotly.graph_objs import Figure
from radstract.data.dicom import (
    DicomTypes,
    convert_dicom_to_images,
    convert_images_to_dicom,
)
from radstract.data.nifti import NIFTI, convert_images_to_nifti_labels

from retuve.classes.draw import Overlay
from retuve.classes.metrics import Metric2D, Metric3D
from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.dev import DevMetricsUS
from retuve.hip_us.classes.general import HipDatasUS, HipDataUS
from retuve.hip_us.draw import draw_hips_us, draw_table
from retuve.hip_us.handlers.bad_data import handle_bad_frames
from retuve.hip_us.handlers.side import set_side_info
from retuve.hip_us.metrics.dev import get_dev_metrics
from retuve.hip_us.modes.landmarks import landmarks_2_metrics_us
from retuve.hip_us.modes.seg import pre_process_segs_us, segs_2_landmarks_us
from retuve.hip_us.multiframe import (
    find_graf_plane,
    get_3d_metrics_and_visuals,
)
from retuve.hip_xray.classes import DevMetricsXRay, HipDataXray, LandmarksXRay
from retuve.hip_xray.draw import draw_hips_xray
from retuve.hip_xray.landmarks import landmarks_2_metrics_xray
from retuve.keyphrases.config import Config, OperationType
from retuve.keyphrases.enums import HipMode
from retuve.logs import ulogger
from retuve.typehints import GeneralModeFuncType


def get_fps(no_of_frames: int, min_fps=30, min_vid_length=6) -> int:
    """
    Get the frames per second for the video clip.

    Should be min_fps or number of fps to produce 6 min_vid_length of video.

    :param no_of_frames: The number of frames.
    :param min_fps: The minimum frames per second.
    :param min_vid_length: The minimum video length.

    :return: The frames per second.
    """

    fps = (
        min_fps
        if no_of_frames > (min_fps * min_vid_length)
        else no_of_frames // min_vid_length
    )

    return fps if fps > 0 else 1


def process_landmarks_xray(
    config: Config,
    landmark_results: List[LandmarksXRay],
    seg_results: List[SegFrameObjects],
) -> Tuple[List[HipDataXray], List[Image.Image]]:
    """
    Process the landmarks for the xray.

    :param config: The configuration.
    :param landmark_results: The landmark results.
    :param seg_results: The segmentation results.

    :return: The hip datas and the image arrays.
    """
    hip_datas_xray = landmarks_2_metrics_xray(landmark_results, config)
    image_arrays = draw_hips_xray(hip_datas_xray, seg_results, config)
    return hip_datas_xray, image_arrays


def process_segs_us(
    config: Config,
    file: BinaryIO,
    modes_func: Callable[
        [BinaryIO, Union[str, Config], Dict[str, Any]],
        List[SegFrameObjects],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
    called_by_2dus: bool = False,
) -> Tuple[HipDatasUS, List[SegFrameObjects], Tuple[int, int, int]]:
    """
    Process the segmentation for the 3DUS.

    :param config: The configuration.
    :param file: The file.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.
    :param called_by_2dus: Whether the calling was from the 2DUS function.

    :return: The hip datas, the results, and the shape.
    """

    results: List[SegFrameObjects] = modes_func(file, config, **modes_func_kwargs_dict)
    results, shape = pre_process_segs_us(results, config)

    # Run any custom segmentation pre-process hooks
    for preprocess in getattr(config.hip, "seg_preprocess_functions", []) or []:
        try:
            name, func = (
                preprocess if isinstance(preprocess, tuple) else (None, preprocess)
            )
            updated = None
            # Try signatures: (results, config), (results,), (results, config, shape)
            for args in [
                (results, config),
                (results,),
                (results, config, shape),
            ]:
                try:
                    updated = func(*args)
                    break
                except TypeError:
                    continue
            if updated is not None:
                results = updated
        except Exception as e:
            ulogger.error(
                f"Seg preprocess function {getattr(preprocess, '__name__', str(preprocess))} failed: {e}"
            )

    pre_edited_results = copy.deepcopy(results)
    landmarks, all_seg_rejection_reasons, ilium_angle_baselines = segs_2_landmarks_us(
        results, config
    )
    pre_edited_landmarks = copy.deepcopy(landmarks)
    hip_datas = landmarks_2_metrics_us(landmarks, shape, config)
    hip_datas.all_seg_rejection_reasons = all_seg_rejection_reasons
    hip_datas.ilium_angle_baselines = ilium_angle_baselines

    # Initialize a container for any custom dev metrics returned by full_metric_functions
    if (
        not hasattr(hip_datas, "dev_metrics_custom")
        or hip_datas.dev_metrics_custom is None
    ):
        hip_datas.dev_metrics_custom = {}

    for metric_name, metric_func in config.hip.full_metric_functions:
        value = 0
        dev_extra = {}
        # Try different call signatures: (hip_datas, results, config), (hip_datas, config), (hip_datas)
        for args in [
            (hip_datas, results, config),
            (hip_datas, config),
            (hip_datas,),
        ]:
            try:
                result = metric_func(*args)
                # Support multiple return shapes: scalar, (scalar, dict), dict-only
                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], dict)
                ):
                    value, dev_extra = result
                elif isinstance(result, dict):
                    dev_extra = result
                    value = 0
                else:
                    value = result
                break
            except TypeError as e:
                if "positional argument" in str(e):
                    continue
                raise e
            except Exception:
                value = 0
                break

        if getattr(hip_datas, "custom_metrics", None) is None:
            hip_datas.custom_metrics = []

        if not called_by_2dus:
            custom_metric = Metric3D(name=metric_name, full=value)
            hip_datas.custom_metrics.append(custom_metric)
        else:
            custom_metric = Metric2D(name=metric_name, value=value)
            hip_datas.custom_metrics.append(custom_metric)

        # Aggregate any custom dev metrics returned by this metric function
        if dev_extra:
            try:
                # Group dev extras by the metric name for clarity
                if metric_name not in hip_datas.dev_metrics_custom:
                    hip_datas.dev_metrics_custom[metric_name] = {}
                hip_datas.dev_metrics_custom[metric_name].update(dev_extra)
            except Exception:
                pass

    if config.test_data_passthrough:
        hip_datas.pre_edited_results = pre_edited_results
        hip_datas.pre_edited_landmarks = pre_edited_landmarks
        hip_datas.pre_edited_hip_datas = copy.deepcopy(hip_datas)

    return hip_datas, results, shape


def analyse_hip_xray_2D(
    img: Union[Image.Image, pydicom.FileDataset],
    keyphrase: Union[str, Config],
    modes_func: Callable[
        [Image.Image, str, Dict[str, Any]],
        Tuple[List[LandmarksXRay], List[SegFrameObjects]],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
) -> Tuple[HipDataXray, Image.Image, DevMetricsXRay]:
    """
    Analyze the hip for the xray.

    :param img: The image.
    :param keyphrase: The keyphrase.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.

    :return: The hip, the image, and the dev metrics.
    """
    if isinstance(keyphrase, str):
        config = Config.get_config(keyphrase)
    else:
        config = keyphrase

    if isinstance(img, pydicom.FileDataset):
        data = img
    elif isinstance(img, Image.Image):
        data = [img]
    else:
        raise ValueError(f"Invalid image type: {type(img)}. Expected Image or DICOM.")

    if config.operation_type in OperationType.LANDMARK:
        landmark_results, seg_results = modes_func(
            data, keyphrase, **modes_func_kwargs_dict
        )
        hip_datas, image_arrays = process_landmarks_xray(
            config, landmark_results, seg_results
        )

    img = image_arrays[0]
    img = Image.fromarray(img)
    hip = hip_datas[0]

    if config.test_data_passthrough:
        hip.seg_results = seg_results

    return hip, img, DevMetricsXRay()


def analyze_synthetic_xray(
    dcm: pydicom.FileDataset,
    keyphrase: Union[str, Config],
    modes_func: Callable[
        [pydicom.FileDataset, str, Dict[str, Any]],
        Tuple[List[LandmarksXRay], List[SegFrameObjects]],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
) -> NIFTI:
    """
    NOTE: Experimental function.

    Useful if the xray images are stacked in a single DICOM file.

    Analyze the hip for the xray.

    :param dcm: The DICOM file.
    :param keyphrase: The keyphrase.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.

    :return: The nifti segmentation file
    """
    if isinstance(keyphrase, str):
        config = Config.get_config(keyphrase)
    else:
        config = keyphrase

    images = convert_dicom_to_images(dcm)
    nifti_frames = []

    try:
        if config.operation_type in OperationType.LANDMARK:
            landmark_results, seg_results = modes_func(
                images, keyphrase, **modes_func_kwargs_dict
            )
            hip_datas, image_arrays = process_landmarks_xray(
                config, landmark_results, seg_results
            )
    except Exception as e:
        ulogger.error(f"Critical Error: {e}")
        return None

    for hip, seg_frame_objs in zip(hip_datas, seg_results):
        shape = seg_frame_objs.img.shape

        overlay = Overlay((shape[0], shape[1], 3), config)
        test = overlay.get_nifti_frame(seg_frame_objs, shape)
        nifti_frames.append(test)

    # Convert to NIfTI
    nifti = convert_images_to_nifti_labels(nifti_frames)

    return nifti


def analyse_hip_3DUS(
    image: Union[pydicom.FileDataset, List[Image.Image]],
    keyphrase: Union[str, Config],
    modes_func: Callable[
        [pydicom.FileDataset, str, Dict[str, Any]],
        List[SegFrameObjects],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
) -> Tuple[
    HipDatasUS,
    ImageSequenceClip,
    Figure,
    Union[DevMetricsXRay, DevMetricsUS],
]:
    """
    Analyze a 3D Ultrasound Hip

    :param dcm: The DICOM file.
    :param keyphrase: The keyphrase.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.

    :return: The hip datas, the video clip, the 3D visual, and the dev metrics.
    """
    start = time.time()

    config = Config.get_config(keyphrase)
    hip_datas = HipDatasUS()

    file_id = modes_func_kwargs_dict.get("file_id")
    if file_id:
        del modes_func_kwargs_dict["file_id"]

    # if a set of images, convert to a DICOM file
    if isinstance(image, list) and all(isinstance(img, Image.Image) for img in image):
        image = convert_images_to_dicom(image)

    try:
        if config.operation_type == OperationType.SEG:
            hip_datas, results, shape = process_segs_us(
                config, image, modes_func, modes_func_kwargs_dict
            )
        elif config.operation_type == OperationType.LANDMARK:
            raise NotImplementedError(
                "This is not yet supported. Please use the seg operation type."
            )
    except Exception as e:
        raise e
        ulogger.error(f"Critical Error: {e}")
        return None, None, None, None

    hip_datas = handle_bad_frames(hip_datas, config)

    if not any(hip.metrics for hip in hip_datas):
        ulogger.error(f"No metrics were found in image.")

    hip_datas.file_id = file_id
    hip_datas = find_graf_plane(hip_datas, results, config=config)

    hip_datas, results = set_side_info(hip_datas, results, config)

    (
        hip_datas,
        visual_3d,
        fem_sph,
        illium_mesh,
        apex_points,
        femoral_sphere,
        avg_normals_data,
        normals_data,
    ) = get_3d_metrics_and_visuals(hip_datas, results, config)

    image_arrays, nifti = draw_hips_us(hip_datas, results, fem_sph, config)

    if config.seg_export:
        hip_datas.nifti = nifti

    hip_datas = get_dev_metrics(hip_datas, results, config)

    # data_image = draw_table(shape, hip_datas)
    # image_arrays.append(data_image)

    ulogger.info(f"Total 3DUS time: {time.time() - start:.2f}s")

    fps = get_fps(
        len(image_arrays),
        config.visuals.min_vid_fps,
        config.visuals.min_vid_length,
    )

    video_clip = ImageSequenceClip(
        image_arrays,
        fps=fps,
    )

    if config.test_data_passthrough:
        hip_datas.illium_mesh = illium_mesh
        hip_datas.fem_sph = fem_sph
        hip_datas.results = results
        hip_datas.apex_points = apex_points
        hip_datas.femoral_sphere = femoral_sphere
        hip_datas.avg_normals_data = avg_normals_data
        hip_datas.normals_data = normals_data

    if getattr(hip_datas, "custom_metrics", None) is not None:
        hip_datas.metrics += hip_datas.custom_metrics

    return (
        hip_datas,
        video_clip,
        visual_3d,
        hip_datas.dev_metrics,
    )


def analyse_hip_2DUS(
    img: Union[Image.Image, pydicom.FileDataset],
    keyphrase: Union[str, Config],
    modes_func: Callable[
        [Image.Image, str, Dict[str, Any]],
        List[SegFrameObjects],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
    return_seg_info: bool = False,
) -> Tuple[HipDataUS, Image.Image, DevMetricsUS]:
    """
    Analyze a 2D Ultrasound Hip

    :param img: The image.
    :param keyphrase: The keyphrase.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.

    :return: The hip, the image, and the dev metrics.
    """
    config = Config.get_config(keyphrase)

    if isinstance(img, pydicom.FileDataset):
        data = img
    elif isinstance(img, Image.Image):
        data = [img]

    try:
        if config.operation_type in OperationType.SEG:
            hip_datas, results, _ = process_segs_us(
                config,
                data,
                modes_func,
                modes_func_kwargs_dict,
                called_by_2dus=True,
            )
    except Exception as e:
        ulogger.error(f"Critical Error: {e}")
        return None, None, None

    image_arrays, _ = draw_hips_us(hip_datas, results, None, config)

    hip_datas = get_dev_metrics(hip_datas, results, config)

    image = image_arrays[0]
    hip = hip_datas[0]

    image = Image.fromarray(image)

    if return_seg_info:
        hip.seg_info = results

    if getattr(hip_datas, "custom_metrics", None) is not None:
        hip.metrics += hip_datas.custom_metrics

    return hip, image, hip_datas.dev_metrics


def analyse_hip_2DUS_sweep(
    image: Union[pydicom.FileDataset, List[Image.Image]],
    keyphrase: Union[str, Config],
    modes_func: Callable[
        [pydicom.FileDataset, str, Dict[str, Any]],
        List[SegFrameObjects],
    ],
    modes_func_kwargs_dict: Dict[str, Any],
) -> Tuple[
    HipDataUS,
    Image.Image,
    DevMetricsUS,
    ImageSequenceClip,
]:
    """
    Analyze a 2D Sweep Ultrasound Hip

    :param dcm: The DICOM file.
    :param keyphrase: The keyphrase.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.

    :return: The hip, the image, the dev metrics, and the video clip.
    """

    config = Config.get_config(keyphrase)
    hip_datas = HipDatasUS()

    # if a set of images, convert to a DICOM file
    if isinstance(image, list) and all(isinstance(img, Image.Image) for img in image):
        image = convert_images_to_dicom(image)

    try:
        if config.operation_type == OperationType.SEG:
            hip_datas, results, shape = process_segs_us(
                config,
                image,
                modes_func,
                modes_func_kwargs_dict,
                called_by_2dus=True,
            )
        elif config.operation_type == OperationType.LANDMARK:
            raise NotImplementedError(
                "This is not yet supported. Please use the seg operation type."
            )
    except Exception as e:
        raise e
        ulogger.error(f"Critical Error: {e}")
        return None, None, None, None

    hip_datas = handle_bad_frames(hip_datas, config)
    hip_datas = find_graf_plane(hip_datas, results, config)

    graf_hip = hip_datas.grafs_hip
    graf_frame = hip_datas.graf_frame

    image_arrays, _ = draw_hips_us(hip_datas, results, None, config)

    hip_datas = get_dev_metrics(hip_datas, results, config)

    if graf_frame is not None:
        graf_image = image_arrays[graf_frame]
        graf_image = Image.fromarray(graf_image)

        image_arrays = (
            [image_arrays[graf_frame]] * int(len(image_arrays) * 0.1)
            + image_arrays
            + [image_arrays[graf_frame]] * int(len(image_arrays) * 0.1)
        )
    else:
        graf_image = None

    video_clip = ImageSequenceClip(
        image_arrays,
        fps=get_fps(
            len(image_arrays),
            config.visuals.min_vid_fps,
            config.visuals.min_vid_length,
        ),
    )

    if (
        graf_hip is None
        or graf_hip.metrics is None
        and hip_datas.custom_metrics is not None
    ):
        graf_hip = hip_datas[0]
        graf_hip.metrics = hip_datas.custom_metrics

    elif getattr(hip_datas, "custom_metrics", None) is not None:
        graf_hip.metrics += hip_datas.custom_metrics

    return graf_hip, graf_image, hip_datas.dev_metrics, video_clip


class RetuveResult:
    """
    The standardised result of the Retuve pipeline.

    :attr hip_datas: The hip datas.
    :attr hip: The hip.
    :attr image: The saved image, if any.
    :attr metrics: The metrics.
    :attr video_clip: The video clip, if any.
    :attr visual_3d: The 3D visual, if any.
    """

    def __init__(
        self,
        metrics: Union[List[Metric2D], List[Metric3D]],
        hip_datas: Union[HipDatasUS, List[HipDataXray]] = None,
        hip: Union[HipDataXray, HipDataUS] = None,
        image: Image.Image = None,
        video_clip: ImageSequenceClip = None,
        visual_3d: Figure = None,
    ):
        self.hip_datas = hip_datas
        self.hip = hip
        self.image = image
        self.metrics = metrics
        self.video_clip = video_clip
        self.visual_3d = visual_3d


def retuve_run(
    hip_mode: HipMode,
    config: Config,
    modes_func: GeneralModeFuncType,
    modes_func_kwargs_dict: Dict[str, Any],
    file: str,
) -> RetuveResult:
    """
    Run the Retuve pipeline with standardised inputs and outputs.

    :param hip_mode: The hip mode.
    :param config: The configuration.
    :param modes_func: The mode function.
    :param modes_func_kwargs_dict: The mode function kwargs.
    :param file: The file.

    :return: The Retuve result standardised output.
    """
    org_file_name = file
    always_dcm = (
        len(config.batch.input_types) == 1 and ".dcm" in config.batch.input_types
    )

    if always_dcm or (file.endswith(".dcm") and ".dcm" in config.batch.input_types):
        file = pydicom.dcmread(file)

    if hip_mode == HipMode.XRAY:
        if not isinstance(file, pydicom.FileDataset):
            file = Image.open(file).convert("RGB")

        hip, image, dev_metrics = analyse_hip_xray_2D(
            file, config, modes_func, modes_func_kwargs_dict
        )
        return RetuveResult(hip.json_dump(config, dev_metrics), image=image, hip=hip)
    elif hip_mode == HipMode.US3D:
        modes_func_kwargs_dict["file_id"] = org_file_name.split("/")[-1]
        hip_datas, video_clip, visual_3d, dev_metrics = analyse_hip_3DUS(
            file, config, modes_func, modes_func_kwargs_dict
        )
        if hip_datas:
            return RetuveResult(
                hip_datas.json_dump(config),
                hip_datas=hip_datas,
                video_clip=video_clip,
                visual_3d=visual_3d,
            )
        else:
            return RetuveResult({})
    elif hip_mode == HipMode.US2D:
        if not isinstance(file, pydicom.FileDataset):
            file = Image.open(file).convert("RGB")
        else:
            file = convert_dicom_to_images(file, dicom_type=DicomTypes.SINGLE)[
                0
            ].convert("RGB")

        hip, image, dev_metrics = analyse_hip_2DUS(
            file, config, modes_func, modes_func_kwargs_dict
        )
        return RetuveResult(hip.json_dump(config, dev_metrics), hip=hip, image=image)
    elif hip_mode == HipMode.US2DSW:
        hip, image, dev_metrics, video_clip = analyse_hip_2DUS_sweep(
            file, config, modes_func, modes_func_kwargs_dict
        )
        json_dump = None
        if hip:
            json_dump = hip.json_dump(config, dev_metrics)

        return RetuveResult(
            json_dump,
            hip=hip,
            image=image,
            video_clip=video_clip,
        )
    else:
        raise ValueError(f"Invalid hip_mode. {hip_mode}")
