from typing import List

from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.enums import Side
from retuve.hip_us.classes.general import Metric3D
from retuve.hip_us.typing import CoordinatesArray3D
from retuve.keyphrases.config import Config
from retuve.logs import ulogger
from retuve.utils import rmean
from retuve.classes.draw import Overlay
import copy
import time
from typing import List

from retuve.classes.draw import Overlay
from retuve.classes.metrics import Metric2D, Metric3D
from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.modes.landmarks import landmarks_2_metrics_us
from retuve.hip_us.modes.seg import segs_2_landmarks_us
from retuve.keyphrases.config import Config
from retuve.logs import ulogger


def get_per_frame_3dus_metrics(hip_datas, config):

    per_frame_funcs = getattr(
        config.hip, "per_frame_metric_functions", []) or []
    custom_names: List[str] = []
    for pf in per_frame_funcs:
        if isinstance(pf, tuple):
            custom_names.append(pf[0])
        else:
            custom_names.append(getattr(pf, "__name__", "custom"))

    for name in custom_names:
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
            if hip_datas.graf_frame is not None:
                graf_value = hip_datas.grafs_hip.get_metric(name)

            hip_datas.metrics.append(
                Metric3D(
                    name=name,
                    graf=graf_value,
                    post=rmean(post_values),
                    ant=rmean(ant_values),
                )
            )

    return hip_datas


def draw_custom(final_hip, final_seg_frame_objs, overlay, config):
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
            out = func(final_hip, final_seg_frame_objs, overlay, config)
            if isinstance(out, Overlay):
                overlay = out
        except Exception as e:
            if config.batch.debug == True:
                raise e
            ulogger.error(
                f"Post-draw function {getattr(func, '__name__', str(func))} failed: {e}"
            )

    return overlay


def get_all_custom_metrics(results: List[SegFrameObjects], shape: CoordinatesArray3D, config: Config, called_by_2dus=False):

    pre_edited_results = None
    pre_edited_landmarks = None

    # Run any custom segmentation pre-process hooks
    for preprocess in getattr(config.hip, "seg_preprocess_functions", []) or []:
        try:
            name, func = (
                preprocess if isinstance(
                    preprocess, tuple) else (None, preprocess)
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
            if config.batch.debug == True:
                raise e
            ulogger.error(
                f"Seg preprocess function {getattr(preprocess, '__name__', str(preprocess))} failed: {e}"
            )

    if config.test_data_passthrough:
        pre_edited_results = copy.deepcopy(results)

    landmarks, all_seg_rejection_reasons, ilium_angle_baselines = segs_2_landmarks_us(
        results, config
    )

    if config.test_data_passthrough:
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

    # Run custom per-frame metrics functions, if any
    per_frame_funcs = getattr(
        config.hip, "per_frame_metric_functions", []) or []
    if per_frame_funcs:
        try:
            for hip, seg_frame_objs in zip(hip_datas, results):
                for pf_name, pf_func in [
                    (
                        (pf[0], pf[1])
                        if isinstance(pf, tuple)
                        else (getattr(pf, "__name__", "custom"), pf)
                    )
                    for pf in per_frame_funcs
                ]:
                    out = pf_func(hip, seg_frame_objs, config)
                    # Expect (value, dev_dict)
                    value, dev_extra = (None, None)
                    if isinstance(out, tuple) and len(out) == 2:
                        value, dev_extra = out
                    else:
                        # Back-compat: allow single numeric
                        value = out
                    if value is not None:
                        try:
                            if hip.metrics is None:
                                hip.metrics = []
                            hip.metrics.append(
                                Metric2D(name=pf_name, value=value))
                        except Exception:
                            pass
                    if isinstance(dev_extra, dict):
                        hip_datas.dev_metrics_custom.setdefault(pf_name, []).append(
                            dev_extra
                        )
        except Exception as e:
            if config.batch.debug == True:
                raise e
            ulogger.error(f"Per-frame metric functions failed: {e}")

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
            except Exception as e:
                if config.batch.debug == True:
                    raise e
                value = 0
                ulogger.error(f"Full metric function failed: {e}")
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

    return hip_datas, pre_edited_results, pre_edited_landmarks


def get_per_frame_xray(hip_datas_xray, seg_results, config):
    # Run custom per-image metric functions (standardized signature)
    per_frame_funcs = getattr(
        config.hip, "per_frame_metric_functions", []) or []
    if per_frame_funcs:
        for hip, seg_frame_objs in zip(hip_datas_xray, seg_results):
            for pf_name, pf_func in [
                (
                    (pf[0], pf[1])
                    if isinstance(pf, tuple)
                    else (getattr(pf, "__name__", "custom"), pf)
                )
                for pf in per_frame_funcs
            ]:
                out = pf_func(hip, seg_frame_objs, config)
                value, dev_extra = (None, None)
                if isinstance(out, tuple) and len(out) == 2:
                    value, dev_extra = out
                else:
                    value = out
                if value is not None:
                    if hip.metrics is None:
                        hip.metrics = []
                    hip.metrics.append(Metric2D(name=pf_name, value=value))

    return hip_datas_xray
