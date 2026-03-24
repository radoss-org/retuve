from typing import List

from retuve.classes.draw import Overlay
from retuve.classes.metrics import Metric2D, Metric3D
from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.enums import Side
from retuve.hip_us.typing import CoordinatesArray3D
from retuve.keyphrases.config import Config
from retuve.logs import ulogger
from retuve.utils import rmean


def _get_hook_list(config, attr_name: str) -> list:
    try:
        hip = config.hip
    except AttributeError:
        return []

    value = hip.__dict__.get(attr_name, None)
    return value or []


def _normalize_hook(hook):
    if isinstance(hook, tuple):
        return hook[0], hook[1]
    return hook.__name__, hook


def _call_with_fallbacks(func, args_options):
    for args in args_options:
        try:
            return func(*args)
        except TypeError as e:
            if "positional argument" in str(e):
                continue
            raise e
    return None


def _append_metric(container, metric):
    if container is None:
        return [metric]
    container.append(metric)
    return container


def draw_custom(final_hip, final_seg_frame_objs, overlay, config):
    post_draw_funcs = _get_hook_list(config, "post_draw_functions")

    for pdf in post_draw_funcs:
        name, func = _normalize_hook(pdf)
        try:
            out = func(final_hip, final_seg_frame_objs, overlay, config)
            if isinstance(out, Overlay):
                overlay = out
        except Exception as e:
            if config.batch.debug == True:
                raise e
            ulogger.error(f"Post-draw function {name} failed: {e}")

    return overlay


def custom_seg_preprocessing(
    results: List[SegFrameObjects],
    shape: CoordinatesArray3D,
    config: Config,
):
    for preprocess in _get_hook_list(config, "seg_preprocess_functions"):
        name, func = _normalize_hook(preprocess)
        try:
            updated = _call_with_fallbacks(
                func,
                [
                    (results, config),
                    (results,),
                    (results, config, shape),
                ],
            )
            if updated is not None:
                results = updated
        except Exception as e:
            if config.batch.debug == True:
                raise e
            ulogger.error(f"Seg preprocess function {name} failed: {e}")

    return results


def _get_metric3d_parts(hip_datas, metric_name: str):
    post_values = [
        hip_data.get_metric(metric_name)
        for hip_data in hip_datas
        if hip_data.side == Side.POST and hip_data.get_metric(metric_name) != 0
    ] or [0]

    ant_values = [
        hip_data.get_metric(metric_name)
        for hip_data in hip_datas
        if hip_data.side == Side.ANT and hip_data.get_metric(metric_name) != 0
    ] or [0]

    graf_value = 0
    if getattr(hip_datas, "graf_frame", None) is not None:
        graf_value = hip_datas.grafs_hip.get_metric(metric_name)

    return {
        "graf": graf_value,
        "post": rmean(post_values),
        "ant": rmean(ant_values),
    }


def get_all_custom_metrics(hip_datas, results, config, called_by_2dus=False):
    per_frame_funcs = _get_hook_list(config, "per_frame_metric_functions")
    if per_frame_funcs:
        try:
            for hip, seg_frame_objs in zip(hip_datas, results):
                for pf in per_frame_funcs:
                    pf_name, pf_func = _normalize_hook(pf)
                    out = pf_func(hip, seg_frame_objs, config)

                    value, dev_extra = (None, None)
                    if isinstance(out, tuple) and len(out) == 2:
                        value, dev_extra = out
                    else:
                        value = out

                    if value is not None:
                        hip.metrics = _append_metric(
                            hip.metrics,
                            Metric2D(name=pf_name, value=value),
                        )

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
        result = _call_with_fallbacks(
            metric_func,
            [
                (hip_datas, results, config),
                (hip_datas, config),
                (hip_datas,),
            ],
        )

        try:
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
        except Exception as e:
            if config.batch.debug == True:
                raise e
            value = 0
            ulogger.error(f"Full metric function failed: {e}")

        if getattr(hip_datas, "custom_metrics", None) is None:
            hip_datas.custom_metrics = []

        if not called_by_2dus:
            parts = _get_metric3d_parts(hip_datas, metric_name)
            custom_metric = Metric3D(
                name=metric_name,
                full=value,
                **parts,
            )
            hip_datas.custom_metrics.append(custom_metric)
        else:
            custom_metric = Metric2D(name=metric_name, value=value)
            hip_datas.custom_metrics.append(custom_metric)

        if dev_extra:
            try:
                if metric_name not in hip_datas.dev_metrics_custom:
                    hip_datas.dev_metrics_custom[metric_name] = {}
                hip_datas.dev_metrics_custom[metric_name].update(dev_extra)
            except Exception:
                pass

    return hip_datas


def get_per_frame_xray(hip_datas_xray, seg_results, config):
    per_frame_funcs = _get_hook_list(config, "per_frame_metric_functions")
    if per_frame_funcs:
        for hip, seg_frame_objs in zip(hip_datas_xray, seg_results):
            for pf in per_frame_funcs:
                pf_name, pf_func = _normalize_hook(pf)
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
