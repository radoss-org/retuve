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
Example: X-ray with a custom per-image metric and custom draw overlay

This example shows how to:
- Define a simple landmark-count metric for an X-ray image
- Register a custom post-draw function to render the count
- Run the X-ray pipeline with manual landmarks from the test dataset

Outputs:
- xray_custom-raw.jpg (raw input)
- xray_custom.jpg (with overlay drawing the custom count)
"""

import json
from PIL import Image

from retuve.defaults.hip_configs import default_xray
from retuve.defaults.manual_seg import manual_predict_xray
from retuve.funcs import analyse_hip_xray_2D
from retuve.testdata import Cases, download_case


# --- Custom metric and draw functions --------------------------------------


def landmark_count_metric(hip, seg_frame_objs, config):
    """Count how many landmarks are present.

    Must return: (value, aux_dict)
    """
    lm = getattr(hip, "landmarks", None)
    if lm is None:
        return 0.0, {"count": 0}
    value = float(sum(1 for v in vars(lm).values() if v is not None))
    return value, {"count": value}


def draw_xray_count(hip, overlay, config):
    """Draw the landmark count onto the image."""
    count = hip.get_metric("landmark count")
    # If our metric didn't run for any reason, fallback to direct counting
    if not count:
        lm = getattr(hip, "landmarks", None)
        count = sum(1 for v in vars(lm).values() if v is not None) if lm else 0
    overlay.draw_text(f"count: {int(count)}", 10, 10, header="h1", grafs=True)
    return overlay


# --- Build a custom config --------------------------------------------------

custom_xray = default_xray.get_copy()
custom_xray.hip.per_frame_metric_functions = [("landmark count", landmark_count_metric)]
custom_xray.hip.post_draw_functions = [("xray count", draw_xray_count)]


# --- Run the pipeline -------------------------------------------------------

img_file, labels_json = download_case(Cases.XRAY_JPG)

img_raw = Image.open(img_file)
labels = json.load(open(labels_json))

hip_data, img_with_overlay, dev_metrics = analyse_hip_xray_2D(
    img_raw,
    keyphrase=custom_xray,
    modes_func=manual_predict_xray,
    modes_func_kwargs_dict=labels,
)

# Save results
img_raw.save("xray_custom-raw.jpg")
img_with_overlay.save("xray_custom.jpg")

# Print metrics (includes ours) for demonstration
print(hip_data.json_dump(custom_xray, dev_metrics))
