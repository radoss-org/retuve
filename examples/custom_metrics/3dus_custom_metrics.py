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
Example: 3D Ultrasound custom full-metric (Graf frame confidence) and Graf-only draw

This example shows how to:
- Define a custom full-metric over the whole 3D study (Graf-frame confidence)
- Reuse a per-frame seg-count metric from 2D to draw only on the Graf frame
- Run the 3DUS pipeline and export the video and 3D HTML

Outputs:
- 3dus_custom.mp4 (video with overlays)
- 3dus_custom.html (3D visualization)
"""

import pydicom

from retuve.defaults.hip_configs import default_US
from retuve.defaults.manual_seg import manual_predict_us_dcm
from retuve.funcs import analyse_hip_3DUS
from retuve.testdata import Cases, download_case


# --- Custom metrics and draw functions -------------------------------------


def seg_object_count_metric(hip, seg_frame_objs, config):
    """Basic per-frame count of segmentation objects."""
    return float(len(seg_frame_objs)), None


def scan_quality_graf(hip_datas, results, config):
    """Compute a simple scan quality as the Graf-frame confidence.

    Returns a single float for the entire 3D study.
    """
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


def draw_seg_count_on_graf(hip, overlay, config):
    """Draw the seg count, but only on the Graf frame."""
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


# --- Build a custom config --------------------------------------------------

custom_3dus = default_US.get_copy()
custom_3dus.hip.full_metric_functions = [
    ("scan quality (graf frame)", scan_quality_graf)
]
custom_3dus.hip.per_frame_metric_functions = [
    ("seg object count", seg_object_count_metric)
]
custom_3dus.hip.post_draw_functions = [
    ("us seg count on graf", draw_seg_count_on_graf)
]


# --- Run the pipeline -------------------------------------------------------

dcm_file, seg_file = download_case(Cases.ULTRASOUND_DICOM)

dcm = pydicom.dcmread(dcm_file)

hip_datas, video_clip, visual_3d, dev_metrics = analyse_hip_3DUS(
    dcm,
    keyphrase=custom_3dus,
    modes_func=manual_predict_us_dcm,
    modes_func_kwargs_dict={"seg": seg_file},
)

# Save results
video_clip.write_videofile("3dus_custom.mp4")
visual_3d.write_html("3dus_custom.html")

# Print full metrics (includes ours) for demonstration
print(hip_datas.json_dump(custom_3dus))
