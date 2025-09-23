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
import numpy as np

import pydicom
from PIL import Image
from radstract.data.dicom import (
    convert_dicom_to_images,
)

from retuve.defaults.hip_configs import (
    test_default_US_custom,
    test_default_xray_custom,
    test_default_3DUS_custom,
)
from retuve.defaults.manual_seg import (
    manual_predict_us,
    manual_predict_us_dcm,
)
from retuve.funcs import (
    analyse_hip_2DUS,
    analyse_hip_xray_2D,
    analyse_hip_3DUS,
)
from retuve.defaults.manual_seg import manual_predict_xray


def assert_nearly_equal_or_show_rgb(
    img_np: np.ndarray, img_2dus_custom_np: np.ndarray, prefix=""
):
    try:
        # MSE difference less than 10
        assert np.mean((img_np - img_2dus_custom_np) ** 2) < 15
    except AssertionError:
        # Ensure uint8 for PIL and clip just in case
        a = img_np
        b = img_2dus_custom_np

        # Assumes shape is (H, W, 3). If not, this will raise, which is fine for debugging.
        img_a_pil = Image.fromarray(a, mode="RGB")
        img_b_pil = Image.fromarray(b, mode="RGB")

        img_a_pil.save(f"{prefix}_img_a.png")
        img_b_pil.save(f"{prefix}_img_b.png")

        raise


def test_generated_custom_draw_2dus_matches_fixture(
    us_file_path, img_2dus_custom_np, expected_us_metrics
):
    """Regenerate the 2DUS image with the custom post-draw hook and compare to fixture image."""
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    images = convert_dicom_to_images(dcm)
    frame_idx = expected_us_metrics["frame_with_results"]

    hip, img_pil, _ = analyse_hip_2DUS(
        images[frame_idx],
        keyphrase=test_default_US_custom,
        modes_func=manual_predict_us,
        modes_func_kwargs_dict={"seg": seg_file, "seg_idx": frame_idx},
    )

    img_np = np.array(img_pil)
    assert img_np.shape == img_2dus_custom_np.shape
    assert_nearly_equal_or_show_rgb(img_np, img_2dus_custom_np, "2dus")


def test_generated_custom_draw_xray_matches_fixture(
    xray_file_path, landmarks_xray, img_xray_custom_np, config_xray
):
    """Regenerate the X-ray image with the custom post-draw hook and compare to fixture image."""

    img = Image.open(xray_file_path)
    hip, img_pil, _ = analyse_hip_xray_2D(
        img,
        keyphrase=test_default_xray_custom,
        modes_func=manual_predict_xray,
        modes_func_kwargs_dict=landmarks_xray,
    )

    img_np = np.array(img_pil)
    assert img_np.shape == img_xray_custom_np.shape
    assert_nearly_equal_or_show_rgb(img_np, img_xray_custom_np, "xray")


def test_generated_custom_draw_3dus_matches_fixture(us_file_path, img_3dus_custom_np):
    """Regenerate the 3DUS graf-frame with the custom post-draw and compare to fixture image."""
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    hip_datas, video_clip, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=test_default_3DUS_custom,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    # Fixture-provided expected image (numpy array)

    # Extract the graf frame
    graf_idx = getattr(hip_datas, "graf_frame", None)
    assert graf_idx is not None, "Graf frame not identified in 3DUS pipeline."

    extracted = None
    for i, frame in enumerate(video_clip.iter_frames(dtype="uint8")):
        if i == graf_idx:
            extracted = frame
            break

    assert (
        extracted is not None
    ), "Failed to extract Graf frame from video clip."
    extracted_np = np.array(extracted)

    assert extracted_np.shape == img_3dus_custom_np.shape
    assert_nearly_equal_or_show_rgb(extracted_np, img_3dus_custom_np, "3dus")
