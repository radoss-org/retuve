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

import copy
from typing import Dict, Tuple

import pydicom
from PIL import Image
from radstract.data.dicom import (
    convert_dicom_to_images,
)

from retuve.defaults.hip_configs import test_default_US
from retuve.defaults.manual_seg import (
    manual_predict_us,
    manual_predict_us_dcm,
)
from retuve.funcs import (
    analyse_hip_2DUS,
    analyse_hip_2DUS_sweep,
    analyse_hip_3DUS,
    analyse_hip_xray_2D,
    retuve_run,
)
from retuve.keyphrases.enums import HipMode
from retuve.hip_us.classes.enums import Side
from retuve.hip_us.draw import draw_hips_us
import numpy as np
from retuve.defaults.hip_configs import default_xray
from retuve.defaults.manual_seg import manual_predict_xray


# --- Simple helper functions for testing ---


def _marker_preprocess(results, config, *args):
    """
    No-op preprocess that drops a marker on the first frame object and returns None.
    Returning None ensures the original list instance is kept (pipeline supports this).
    """
    if results:
        try:
            setattr(results[0], "_preprocess_marker", True)
        except Exception:
            pass
    # Explicitly return None to keep current results
    return None


def _simple_count_metric(hip_datas, results, config) -> Tuple[int, Dict]:
    """Return number of frames and echo back whether the preprocess marker was set."""
    marker = False
    try:
        marker = (
            bool(getattr(results[0], "_preprocess_marker", False)) if results else False
        )
    except Exception:
        marker = False
    return len(results or []), {"preprocess_marker_seen": marker}


def _constant_metric(hip_datas) -> int:
    """A constant metric to verify alternate callable signatures are accepted."""
    return 42


def _make_test_config():
    cfg = test_default_US.get_copy()
    # Register our simple custom metrics and preprocess hook
    cfg.hip.full_metric_functions = [
        ("simple count", _simple_count_metric),
        ("constant answer", _constant_metric),
    ]
    cfg.hip.seg_preprocess_functions = [
        ("marker preprocess", _marker_preprocess),
    ]
    return cfg


# --- Per-frame metric and post-draw helpers ---


def _seg_object_count(hip, seg_frame_objs, config):
    """Simple per-frame metric: count number of segment objects."""
    try:
        return float(len(seg_frame_objs)), None
    except Exception:
        return 0.0, None


def _draw_graf_marker(hip, overlay, config):
    """
    Post-draw hook: on the Graf frame, draw a small cross at a fixed position.
    Makes it easy to detect in pixel comparisons.
    """
    try:
        if getattr(hip, "side", None) == Side.GRAF:
            # draw a small cross near the top-left to minimise clashes with other drawings
            overlay.draw_cross((10, 10))
    except Exception:
        pass
    return overlay


def _get_frame_count_from_dcm(us_file_path: str) -> int:
    dcm = pydicom.dcmread(us_file_path)
    images = convert_dicom_to_images(dcm)
    return len(images)


# --- Tests ---


def test_custom_full_metrics_through_analyse_hip_3DUS(us_file_path):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")

    cfg = _make_test_config()

    dcm = pydicom.dcmread(us_file_path)
    hip_datas, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=cfg,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    # Collect custom metrics by name
    by_name = {m.name: m for m in (hip_datas.custom_metrics or [])}

    # simple count equals number of frames
    expected_count = _get_frame_count_from_dcm(us_file_path)
    assert "simple count" in by_name
    assert by_name["simple count"].full == expected_count

    # constant metric present
    assert "constant answer" in by_name
    assert by_name["constant answer"].full == 42

    # dev extras grouped under metric name and contain preprocess marker flag
    assert getattr(hip_datas, "dev_metrics_custom", None) is not None
    assert "simple count" in hip_datas.dev_metrics_custom
    assert (
        hip_datas.dev_metrics_custom["simple count"].get("preprocess_marker_seen")
        is True
    )


def test_per_frame_metric_counts_and_storage(us_file_path):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")

    cfg = _make_test_config()
    # Register a per-frame metric function
    cfg.hip.per_frame_metric_functions = [("seg object count", _seg_object_count)]

    dcm = pydicom.dcmread(us_file_path)
    hip_datas, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=cfg,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    # Expect that each hip frame now has a metric named "seg object count"
    assert len(hip_datas) > 0
    assert (
        hasattr(hip_datas, "pre_edited_results")
        and hip_datas.pre_edited_results is not None
    )

    for hip, seg_frame_objs in zip(hip_datas, hip_datas.pre_edited_results):
        val = hip.get_metric("seg object count")
        # Value should equal number of seg objects for that frame
        assert isinstance(val, (int, float))
        assert val == float(len(seg_frame_objs))


def test_custom_full_metrics_through_retuve_run_3D(us_file_path):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")

    cfg = _make_test_config()

    result = retuve_run(
        hip_mode=HipMode.US3D,
        config=cfg,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
        file=us_file_path,
    )

    hip_datas = result.hip_datas
    by_name = {m.name: m for m in (hip_datas.custom_metrics or [])}

    expected_count = _get_frame_count_from_dcm(us_file_path)
    assert by_name["simple count"].full == expected_count
    assert by_name["constant answer"].full == 42

    assert getattr(hip_datas, "dev_metrics_custom", None) is not None
    assert (
        hip_datas.dev_metrics_custom["simple count"].get("preprocess_marker_seen")
        is True
    )


def test_custom_full_metrics_through_analyse_hip_2DUS(
    us_file_path, expected_us_metrics
):
    # Use a single frame image from the DICOM
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)
    images = convert_dicom_to_images(dcm)
    frame_idx = expected_us_metrics["frame_with_results"]

    cfg = _make_test_config()

    hip, img, _ = analyse_hip_2DUS(
        images[frame_idx],
        keyphrase=cfg,
        modes_func=manual_predict_us,
        modes_func_kwargs_dict={"seg": seg_file, "seg_idx": frame_idx},
    )

    # In 2D, custom metrics are appended onto the hip.metrics as Metric2D(name, value)
    by_name = {m.name: m for m in (hip.metrics or [])}

    assert "simple count" in by_name and by_name["simple count"].value == 1
    assert "constant answer" in by_name and by_name["constant answer"].value == 42


def test_post_draw_hook_renders_on_graf_and_matches_draw_hips(us_file_path):
    """
    Register a post-draw hook that draws a small cross on the Graf frame.
    Compare the PIL image returned by analyse_hip_2DUS_sweep with the
    corresponding NumPy frame produced directly by draw_hips_us.
    Expect exact pixel match.
    """
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    cfg = _make_test_config()
    cfg.hip.post_draw_functions = [("graf marker", _draw_graf_marker)]

    # Run 2D sweep which should render the Graf frame as a PIL image
    hip_datas, img_pil, dev_metrics, _ = analyse_hip_2DUS_sweep(
        dcm,
        keyphrase=cfg,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    # Re-render all frames via draw_hips_us (no fem_sph in 2D sweep)
    assert (
        hasattr(hip_datas, "pre_edited_results")
        and hip_datas.pre_edited_results is not None
    )
    images, _ = draw_hips_us(hip_datas, hip_datas.pre_edited_results, None, cfg)

    graf_idx = hip_datas.graf_frame
    img_np_from_draw = images[graf_idx]
    img_np_from_sweep = np.array(img_pil)

    assert img_np_from_draw.shape == img_np_from_sweep.shape
    assert np.array_equal(img_np_from_draw, img_np_from_sweep)


def test_generated_custom_draw_2dus_matches_fixture(
    us_file_path, img_2dus_custom_np, expected_us_metrics
):
    """Regenerate the 2DUS image with the custom post-draw hook and compare to fixture image."""
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    cfg = _make_test_config()

    # Same helpers used by test data generation
    def _seg_object_count(hip, seg_frame_objs, config):
        try:
            return float(len(seg_frame_objs)), None
        except Exception:
            return 0.0, None

    def _draw_us_seg_count_on_graf(hip, overlay, config):
        try:
            if getattr(hip, "side", None) == Side.GRAF:
                count = hip.get_metric("seg object count") or 0
                overlay.draw_text(
                    f"count: {int(count)}", 20, 20, header="h2", grafs=True
                )
        except Exception:
            pass
        return overlay

    cfg.hip.per_frame_metric_functions = [("seg object count", _seg_object_count)]
    cfg.hip.post_draw_functions = [("us seg count on graf", _draw_us_seg_count_on_graf)]

    images = convert_dicom_to_images(dcm)
    frame_idx = expected_us_metrics["frame_with_results"]

    hip, img_pil, _ = analyse_hip_2DUS(
        images[frame_idx],
        keyphrase=cfg,
        modes_func=manual_predict_us,
        modes_func_kwargs_dict={"seg": seg_file, "seg_idx": frame_idx},
    )

    img_np = np.array(img_pil)
    assert img_np.shape == img_2dus_custom_np.shape
    assert np.array_equal(img_np, img_2dus_custom_np)


def test_generated_custom_draw_xray_matches_fixture(
    xray_file_path, landmarks_xray, img_xray_custom_np
):
    """Regenerate the X-ray image with the custom post-draw hook and compare to fixture image."""

    def _draw_xray_custom(hip, overlay, config):
        try:
            val = hip.get_metric("ihdi") if hasattr(hip, "get_metric") else None
            txt = f"custom: {val if val is not None else 0}"
            overlay.draw_text(txt, 20, 20, header="h2")
        except Exception:
            pass
        return overlay

    cfg = default_xray.get_copy()
    cfg.hip.post_draw_functions = [("xray custom", _draw_xray_custom)]

    img = Image.open(xray_file_path)
    hip, img_pil, _ = analyse_hip_xray_2D(
        img,
        keyphrase=cfg,
        modes_func=manual_predict_xray,
        modes_func_kwargs_dict=landmarks_xray,
    )

    img_np = np.array(img_pil)
    assert img_np.shape == img_xray_custom_np.shape
    assert np.array_equal(img_np, img_xray_custom_np)


def test_custom_full_metrics_through_analyse_hip_2DUS_sweep(us_file_path):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    cfg = _make_test_config()

    dcm = pydicom.dcmread(us_file_path)

    hip, img, _, _ = analyse_hip_2DUS_sweep(
        dcm,
        keyphrase=cfg,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    # In 2D sweep, custom metrics are appended onto the graf hip's metrics
    by_name = {m.name: m for m in (hip.metrics or [])}

    expected_count = _get_frame_count_from_dcm(us_file_path)
    assert "simple count" in by_name and by_name["simple count"].value == expected_count
    assert "constant answer" in by_name and by_name["constant answer"].value == 42
