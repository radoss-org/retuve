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
Example: 2D Ultrasound with a custom per-frame metric and custom draw overlay

This example shows how to:
- Define a simple per-frame metric (count segment objects)
- Register a custom post-draw function that renders that metric on the image
- Run the 2DUS pipeline with manual labels from the test dataset

Outputs:
- 2dus_custom-raw.png (the raw frame)
- 2dus_custom.png (with overlay that includes our custom draw)
"""

import pydicom
from PIL import Image
from radstract.data.dicom import convert_dicom_to_images

from retuve.defaults.hip_configs import default_US
from retuve.defaults.manual_seg import manual_predict_us
from retuve.funcs import analyse_hip_2DUS
from retuve.testdata import Cases, download_case


# --- Custom metric and draw functions --------------------------------------


def seg_object_count_metric(hip, seg_frame_objs, config):
    """Return how many segmentation objects exist on this frame.

    Must return: (value, aux_dict)
    """
    return float(len(seg_frame_objs)), None


def draw_seg_count(hip, overlay, config):
    """Draw the per-frame "seg object count" on the image."""
    count = hip.get_metric("seg object count")
    overlay.draw_text(f"count: {int(count or 0)}", 50, 50, header="h2")
    return overlay


# --- Build a custom config --------------------------------------------------

custom_us = default_US.get_copy()
custom_us.hip.per_frame_metric_functions = [
    ("seg object count", seg_object_count_metric)
]
custom_us.hip.post_draw_functions = [("seg count", draw_seg_count)]


# --- Run the pipeline -------------------------------------------------------

dcm_file, seg_file = download_case(Cases.ULTRASOUND_DICOM)

dcm = pydicom.dcmread(dcm_file)
images = convert_dicom_to_images(dcm)

# Use the first frame for a simple still example
raw_frame: Image.Image = images[0]

hip_data, img_with_overlay, dev_metrics = analyse_hip_2DUS(
    raw_frame,
    keyphrase=custom_us,
    modes_func=manual_predict_us,
    modes_func_kwargs_dict={"seg": seg_file},
)

# Save results
raw_frame.save("2dus_custom-raw.png")
img_with_overlay.save("2dus_custom.png")

# Print metrics (includes ours) for demonstration
print(hip_data.json_dump(custom_us, dev_metrics))
