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
Default Working Configs for Retuve

Import them directly from the retuve package.

We don't store them, so that the Retuve UI does not use them.
"""

from radstract.data.dicom import DicomTypes

from retuve.keyphrases.enums import MetricUS, OperationType

from .config import base_config

default_US = base_config.get_copy()
default_US.visuals.default_font_size = 20
default_US.visuals.points_radius = 10
default_US.visuals.line_thickness = 3
default_US.visuals.bounding_box_thickness = 8
default_US.hip.display_side = False
default_US.visuals.display_segs = True
default_US.batch.input_types = [".dcm", ".jpg", ".png"]

default_US.register(name="default_US", store=False, silent=True)

default_xray = base_config.get_copy()
default_xray.operation_type = OperationType.LANDMARK
default_xray.visuals.points_radius = 10
default_xray.visuals.line_thickness = 3
default_xray.visuals.default_font_size = 20
default_xray.visuals.bounding_box_thickness = 7
default_xray.dicom_type = DicomTypes.SINGLE
default_xray.batch.input_types = [".dcm", ".jpg", ".png"]

default_xray.register(name="default_xray", store=False, silent=True)

live = default_US.get_copy()

test_default_US = default_US.get_copy()
test_default_US.hip.measurements = [
    MetricUS.ALPHA,
    MetricUS.COVERAGE,
    MetricUS.CENTERING_RATIO,
    MetricUS.ACA,
]

test_default_US_custom = test_default_US.get_copy()


# Same helpers used by test data generation
def _seg_object_count(hip, seg_frame_objs, config):
    return float(len(seg_frame_objs)), None


def _draw_us_seg_count(hip, overlay, config):
    count = hip.get_metric("seg object count")
    overlay.draw_text(f"count: {count}", 50, 50, header="h2")


test_default_US_custom.hip.per_frame_metric_functions = [
    ("seg object count", _seg_object_count)
]
test_default_US_custom.hip.post_draw_functions = [
    ("us seg count", _draw_us_seg_count)
]

test_default_xray_custom = default_xray.get_copy()


# Example custom per-image metric for XRAY
def landmark_count_metric(hip, seg_frame_objs, config):
    try:
        lm = getattr(hip, "landmarks", None)
        if lm is None:
            return None, {}
        value = float(sum(1 for v in vars(lm).values() if v is not None))
        return value, {"count": value}
    except Exception:
        return None, {}


# Example custom post-draw function for XRAY: draw landmark count
def draw_xray_count(hip, overlay, config):
    try:
        count = hip.get_metric("landmark count")
        if not count:
            lm = getattr(hip, "landmarks", None)
            count = (
                sum(1 for v in vars(lm).values() if v is not None) if lm else 0
            )
        # Use graf colors for stronger contrast on x-ray
        overlay.draw_text(
            f"count: {int(count)}", 10, 10, header="h1", grafs=True
        )
    except Exception:
        pass
    return overlay


test_default_xray_custom.hip.per_frame_metric_functions = [
    ("landmark count", landmark_count_metric),
]
test_default_xray_custom.hip.post_draw_functions = [
    ("xray count", draw_xray_count),
]

# -----------------------------------------------------------------------------
# 3DUS: Example custom full metric and graf-frame drawing
# -----------------------------------------------------------------------------

test_default_3DUS_custom = default_US.get_copy()


def _scan_quality_graf(hip_datas, results, config):
    """Return the Graf-frame confidence as a single "full" metric."""
    try:
        from retuve.hip_us.multiframe import find_graf_plane

        hip_datas = find_graf_plane(hip_datas, results, config)
        if getattr(hip_datas, "graf_frame", None) is None or not getattr(
            hip_datas, "graf_confs", []
        ):
            return 0.0
        return float(round(hip_datas.graf_confs[hip_datas.graf_frame], 2))
    except Exception:
        return 0.0


def _draw_seg_count_on_graf(hip, overlay, config):
    """Draw a simple counter only on the Graf frame.

    Reuses the per-frame metric from the US example ("seg object count").
    """
    try:
        from retuve.hip_us.classes.enums import Side

        if getattr(hip, "side", None) != Side.GRAF:
            return overlay
        count = hip.get_metric("seg object count") or 0
        overlay.draw_text(
            f"count: {int(count)}", 20, 20, header="h2", grafs=True
        )
    except Exception:
        pass
    return overlay


# Register full-metric and post-draw for 3DUS custom config
test_default_3DUS_custom.hip.full_metric_functions = [
    ("scan quality (graf frame)", _scan_quality_graf)
]

# Reuse the per-frame seg count metric defined above for US
test_default_3DUS_custom.hip.per_frame_metric_functions = [
    ("seg object count", _seg_object_count)
]

test_default_3DUS_custom.hip.post_draw_functions = [
    ("us seg count on graf", _draw_seg_count_on_graf)
]
