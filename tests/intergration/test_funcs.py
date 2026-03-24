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

import os

import pydicom
from PIL import Image
from radstract.data.dicom import (
    convert_dicom_to_images,
    convert_images_to_dicom,
)

from retuve.defaults.hip_configs import default_xray, test_default_US
from retuve.defaults.manual_seg import (
    manual_predict_us,
    manual_predict_us_dcm,
    manual_predict_xray,
)
from retuve.funcs import (
    analyse_hip_2DUS,
    analyse_hip_2DUS_sweep,
    analyse_hip_3DUS,
    analyse_hip_xray_2D,
    retuve_run,
)
from retuve.keyphrases.enums import HipMode


def test_analyse_hip_3DUS(us_file_path, metrics_3d_us):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")

    dcm = pydicom.dcmread(us_file_path)

    hip_datas, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=test_default_US,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    metrics = hip_datas.json_dump(test_default_US)
    del metrics["recorded_error"]
    if "recorded_error" in metrics_3d_us:
        del metrics_3d_us["recorded_error"]

    assert metrics == metrics_3d_us


def test_retuve_run_3DUS(us_file_path, metrics_3d_us):
    seg_file = us_file_path.replace(".dcm", ".nii.gz")

    retuve_result = retuve_run(
        hip_mode=HipMode.US3D,
        config=test_default_US,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
        file=us_file_path,
    )
    del retuve_result.metrics["recorded_error"]
    if "recorded_error" in metrics_3d_us:
        del metrics_3d_us["recorded_error"]

    assert retuve_result.metrics == metrics_3d_us


def test_analyse_hip_xray_2D(landmarks_xray, xray_file_path, metrics_xray):
    img = Image.open(xray_file_path)

    hip_data, img, dev_metrics = analyse_hip_xray_2D(
        img,
        keyphrase=default_xray,
        modes_func=manual_predict_xray,
        modes_func_kwargs_dict=landmarks_xray,
    )

    metrics = hip_data.json_dump(default_xray, dev_metrics)

    assert metrics == metrics_xray


def test_retuve_run_xray(landmarks_xray, xray_file_path, metrics_xray):

    retuve_result = retuve_run(
        hip_mode=HipMode.XRAY,
        config=default_xray,
        modes_func=manual_predict_xray,
        modes_func_kwargs_dict=landmarks_xray,
        file=xray_file_path,
    )

    # Only added in retuve-run presently
    del retuve_result.metrics["landmarks"]

    assert retuve_result.metrics == metrics_xray


def test_analyse_hip_2DUS(us_file_path, metrics_2d_us, expected_us_metrics):

    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    images = convert_dicom_to_images(dcm)

    hip_data, _, dev_metrics = analyse_hip_2DUS(
        images[expected_us_metrics["frame_with_results"]],
        keyphrase=test_default_US,
        modes_func=manual_predict_us,
        modes_func_kwargs_dict={
            "seg": seg_file,
            "seg_idx": expected_us_metrics["frame_with_results"],
        },
    )

    metrics = hip_data.json_dump(test_default_US, dev_metrics)

    assert metrics == metrics_2d_us


def test_analyse_hip_2DUS_sweep(us_file_path, metrics_2d_sweep):

    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)

    hip_data, _, dev_metrics, _ = analyse_hip_2DUS_sweep(
        dcm,
        keyphrase=test_default_US,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    metrics = hip_data.json_dump(test_default_US, dev_metrics)

    assert metrics == metrics_2d_sweep


def test_analyse_hip_3DUS_plain_images(us_file_path, metrics_3d_us):
    """Test 3DUS analysis with a list of PIL Images."""
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)
    images = convert_dicom_to_images(dcm)

    hip_datas, *_ = analyse_hip_3DUS(
        images,
        keyphrase=test_default_US,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    metrics = hip_datas.json_dump(test_default_US)
    del metrics["recorded_error"]
    if "recorded_error" in metrics_3d_us:
        del metrics_3d_us["recorded_error"]

    assert metrics == metrics_3d_us


def test_analyse_hip_2DUS_sweep_plain_images(us_file_path, metrics_2d_sweep):
    """Test 2D sweep analysis with a list of PIL Images."""
    seg_file = us_file_path.replace(".dcm", ".nii.gz")
    dcm = pydicom.dcmread(us_file_path)
    images = convert_dicom_to_images(dcm)

    hip_data, _, dev_metrics, _ = analyse_hip_2DUS_sweep(
        images,
        keyphrase=test_default_US,
        modes_func=manual_predict_us_dcm,
        modes_func_kwargs_dict={"seg": seg_file},
    )

    metrics = hip_data.json_dump(test_default_US, dev_metrics)
    assert metrics == metrics_2d_sweep
