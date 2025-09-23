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

import gzip
import json
import os
import pickle
import shutil
from datetime import datetime
from difflib import unified_diff

import open3d as o3d
import pydicom
import yaml
from PIL import Image
from radstract.data.dicom import convert_dicom_to_images

from retuve.defaults.hip_configs import default_xray, test_default_US
from retuve.hip_us.classes.enums import Side
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
)
from retuve.testdata import Cases, download_case
from retuve.defaults.hip_configs import (
    test_default_US_custom,
    test_default_xray_custom,
    test_default_3DUS_custom,
)

# DISCLAIMER
# =======================
# Before running this code, please read the following disclaimer carefully:

print(
    """
DISCLAIMER
=======================
Before running the test generation script, please read the following disclaimer carefully:

The data used by these tests is licensed under the CC BY-NC-SA 3.0 license, as detailed here:
https://github.com/radoss-org/radoss-creative-commons

By running tests with this data, you agree to abide by the terms of the CC BY-NC-SA 3.0 license:
- You may share and adapt the data, but you must give appropriate credit, provide a link to the license, and indicate if changes were made.
- You may not use the data for commercial purposes.
- If you remix, transform, or build upon the data, you must distribute your contributions under the same license.

If you do not agree to these terms, do not proceed with running these tests.

Please type "yes" to confirm that you have read and agree to the terms of the CC BY-NC-SA 3.0 license.
"""
)

if os.environ.get("RETUVE_DISABLE_WARNING").lower() != "true":
    user_input = input(
        "Do you agree to the terms of the CC BY-NC-SA 3.0 license? (Type 'yes' to continue): "
    )
else:
    user_input = "yes"

if user_input.lower() != "yes":
    print("You did not agree to the terms. Exiting...")
    exit()

print("Thank you for agreeing to the terms. Proceeding with the test generation...")

# remove if exists and create test-data dir
if os.path.exists("./tests/test-data"):
    shutil.rmtree("./tests/test-data")

os.makedirs("./tests/test-data", exist_ok=True)

# Example usage
dcm_file, seg_file = download_case(
    Cases.ULTRASOUND_DICOM, directory="./tests/test-data"
)

dcm = pydicom.dcmread(dcm_file)

dcm = pydicom.dcmread(dcm_file)
test_default_US.test_data_passthrough = True

hip_datas, video_clip, visual_3d, dev_metrics = analyse_hip_3DUS(
    dcm,
    keyphrase=test_default_US,
    modes_func=manual_predict_us_dcm,
    modes_func_kwargs_dict={"seg": seg_file},
)

illium_mesh = hip_datas.illium_mesh
fem_sph = hip_datas.fem_sph
results = hip_datas.results
apex_points = hip_datas.apex_points
femoral_sphere = hip_datas.femoral_sphere
avg_normals_data = hip_datas.avg_normals_data
normals_data = hip_datas.normals_data
pre_edited_hip_datas = hip_datas.pre_edited_hip_datas
pre_edited_results = hip_datas.pre_edited_results
pre_edited_landmarks = hip_datas.pre_edited_landmarks

# make these elements none in hip_datas now that they are passed through
hip_datas.illium_mesh = None
hip_datas.fem_sph = None
hip_datas.results = None
hip_datas.apex_points = None
hip_datas.femoral_sphere = None
hip_datas.avg_normals_data = None
hip_datas.normals_data = None
hip_datas.pre_edited_hip_datas = None
hip_datas.pre_edited_results = None
hip_datas.pre_edited_landmarks = None

test_data_dir = "./tests/test-data"

# Save the mesh using open3d
o3d.io.write_triangle_mesh(
    f"{test_data_dir}/illium_mesh.obj", illium_mesh, write_ascii=True
)

# Save json
json_file_3dus = hip_datas.json_dump(test_default_US)

# Save the data
with gzip.open(f"{test_data_dir}/hip_datas_us.pkl.gz", "wb") as f:
    pickle.dump(hip_datas, f)

# Save the mesh as a
with gzip.open(f"{test_data_dir}/fem_sph.pkl.gz", "wb") as f:
    pickle.dump(fem_sph, f)

with gzip.open(f"{test_data_dir}/results_us.pkl.gz", "wb") as f:
    pickle.dump(results, f)

# store apex_points
with gzip.open(f"{test_data_dir}/apex_points.pkl.gz", "wb") as f:
    pickle.dump(apex_points, f)

with gzip.open(f"{test_data_dir}/femoral_sphere.pkl.gz", "wb") as f:
    pickle.dump(femoral_sphere, f)

with gzip.open(f"{test_data_dir}/avg_normals_data.pkl.gz", "wb") as f:
    pickle.dump(avg_normals_data, f)

with gzip.open(f"{test_data_dir}/normals_data.pkl.gz", "wb") as f:
    pickle.dump(normals_data, f)


with gzip.open(f"{test_data_dir}/pre_edited_hip_datas.pkl.gz", "wb") as f:
    pickle.dump(pre_edited_hip_datas, f)

with gzip.open(f"{test_data_dir}/pre_edited_results.pkl.gz", "wb") as f:
    pickle.dump(pre_edited_results, f)

with gzip.open(f"{test_data_dir}/pre_edited_landmarks.pkl.gz", "wb") as f:
    pickle.dump(pre_edited_landmarks, f)

# Create visuals to ensure it works
video_clip.write_videofile(f"{test_data_dir}/video_us.mp4")

visual_3d.write_html(f"{test_data_dir}/visual_3d_us.html")


# Example usage
img_file, labels_json = download_case(Cases.XRAY_JPG, directory="./tests/test-data")

img_raw = Image.open(img_file)
labels = json.load(open(labels_json))
default_xray.test_data_passthrough = True

hip_data, img, dev_metrics = analyse_hip_xray_2D(
    img_raw,
    keyphrase=default_xray,
    modes_func=manual_predict_xray,
    modes_func_kwargs_dict=labels,
)

# Save json
json_file_xray = hip_data.json_dump(default_xray, dev_metrics)

img.save(f"{test_data_dir}/img_xray.jpg")

seg_results = hip_data.seg_results

hip_data.seg_results = None

# Save the data
with gzip.open(f"{test_data_dir}/hip_data_xray.pkl.gz", "wb") as f:
    pickle.dump(hip_data, f)

with gzip.open(f"{test_data_dir}/seg_results_xray.pkl.gz", "wb") as f:
    pickle.dump(seg_results, f)


# Get jsons for 2DSW and 2D

hip_datas, img, dev_metrics, _ = analyse_hip_2DUS_sweep(
    dcm,
    keyphrase=test_default_US,
    modes_func=manual_predict_us_dcm,
    modes_func_kwargs_dict={"seg": seg_file},
)

# save 2d image
img.save(f"{test_data_dir}/img_2dus_sweep.jpg")

json_file_us_sweep = hip_datas.json_dump(test_default_US, dev_metrics)

images = convert_dicom_to_images(dcm)

FRAME = 4

hip_data, img, dev_metrics = analyse_hip_2DUS(
    img=images[FRAME],
    keyphrase=test_default_US,
    modes_func=manual_predict_us,
    modes_func_kwargs_dict={"seg": seg_file, "seg_idx": FRAME},
)

# save 2d image
img.save(f"{test_data_dir}/img_2dus.jpg")

json_file_us = hip_data.json_dump(test_default_US, dev_metrics)


# ------------------------------------------------------------------------------
# Generate 2DUS and X-ray images with a custom post-draw overlay
# ------------------------------------------------------------------------------


hip_custom, img_custom, dev_custom = analyse_hip_2DUS(
    img=images[FRAME],
    keyphrase=test_default_US_custom,
    modes_func=manual_predict_us,
    modes_func_kwargs_dict={"seg": seg_file, "seg_idx": FRAME},
)

img_custom.save(f"{test_data_dir}/img_2dus_custom.jpg")


# X-ray: add a post-draw hook that writes a simple, deterministic label.
def _draw_xray_custom(hip, overlay, config):
    try:
        val = hip.get_metric("ihdi") if hasattr(hip, "get_metric") else None
        txt = f"custom: {val if val is not None else 0}"
        overlay.draw_text(txt, 20, 20, header="h2")
    except Exception:
        pass
    return overlay


hip_xc, img_xc, dev_xc = analyse_hip_xray_2D(
    img_raw,
    keyphrase=test_default_xray_custom,
    modes_func=manual_predict_xray,
    modes_func_kwargs_dict=labels,
)

img_xc.save(f"{test_data_dir}/img_xray_custom.jpg")


# ==============================================================================

# 3DUS custom: generate a graf-frame image using the custom config

hip_c3d, video_c3d, visual3d_c3d, dev_c3d = analyse_hip_3DUS(
    dcm,
    keyphrase=test_default_3DUS_custom,
    modes_func=manual_predict_us_dcm,
    modes_func_kwargs_dict={"seg": seg_file},
)

# Extract and save the Graf frame from the generated video
try:
    graf_idx = getattr(hip_c3d, "graf_frame", None)
    if graf_idx is not None:
        # Iterate frames to the Graf index and save that frame
        for i, frame in enumerate(video_c3d.iter_frames(dtype="uint8")):
            if i == graf_idx:
                Image.fromarray(frame).save(f"{test_data_dir}/img_3dus_custom.jpg")
                break
except Exception:
    pass


def load_previous_release_note():
    # Locate the most recent YAML file in ./changenotes/
    release_dir = "./changenotes/"
    yaml_files = [f for f in os.listdir(release_dir) if f.endswith(".yaml")]
    # sort by date (most recent first)
    yaml_files.sort(reverse=True)
    if yaml_files:
        with open(os.path.join(release_dir, yaml_files[0]), "r") as f:
            return yaml.safe_load(f), yaml_files[0]
    return None, None


def has_changes(new_data, old_data):
    # Change recorded error to None in both if "" or None
    if new_data["3dus"]["recorded_error"] in ["", None]:
        new_data["3dus"]["recorded_error"] = None
    if old_data["3dus"]["recorded_error"] in ["", None]:
        old_data["3dus"]["recorded_error"] = None

    # Compare for changes as JSON's
    changes = json.dumps(new_data, sort_keys=True) != json.dumps(
        old_data, sort_keys=True
    )

    # Print diff
    if changes:
        diff = unified_diff(
            json.dumps(old_data, indent=4).splitlines(),
            json.dumps(new_data, indent=4).splitlines(),
            lineterm="",
        )
        print("\n".join(diff))
    return changes


# Helper function to add comments for modified YAML values
def mark_changes(new_data, old_data, path=""):
    if isinstance(new_data, dict):
        # Recursively compare dictionaries
        for key in new_data.keys():
            full_path = f"{path}.{key}" if path else key
            if key in old_data:
                new_data[key] = mark_changes(new_data[key], old_data[key], full_path)
            else:
                # Key is new, mark it as added
                new_data[key] = f"{new_data[key]}  # Added"
    elif isinstance(new_data, list):
        # Compare lists by index
        for i in range(len(new_data)):
            if i < len(old_data):
                new_data[i] = mark_changes(new_data[i], old_data[i], f"{path}[{i}]")
            else:
                # Item is new in the list
                new_data[i] = f"{new_data[i]}  # Added"
    else:
        # Direct value comparison for basic data types
        if new_data != old_data:
            # Mark value as modified
            new_data = f"{new_data}  # Modified"
    return new_data


# Load previous release note
previous_data, previous_filename = load_previous_release_note()

# Merge the json files, save as YAML
merged_json = {
    "description": "Your description here.",
    "current_output": {
        "3dus": json_file_3dus,
        "xray": json_file_xray,
        "us_sweep": json_file_us_sweep,
        "2dus": json_file_us,
    },
}

json_str = json.dumps(merged_json, default=str)
standardized_data = json.loads(json_str)

# Check for changes
if previous_data is None or has_changes(
    standardized_data["current_output"], previous_data["current_output"]
):
    if previous_data:
        new_current_output = mark_changes(
            standardized_data["current_output"],
            previous_data["current_output"],
        )
        annotated_data = {
            "description": standardized_data["description"],
            "current_output": new_current_output,
        }
    else:
        annotated_data = standardized_data

    # Prepare filename and save as YAML
    today = datetime.today().strftime("%Y-%m-%d")
    new_filename = f"./changenotes/{today}_title-here.yaml"

    with open(new_filename, "w") as f:
        yaml.dump(annotated_data, f, sort_keys=False)

    # Modify values with # Modified, by removing all quotes
    with open(new_filename, "r") as f:
        content = f.read()
        content = content.replace("'", "")
        content = content.replace("recorded_error:   # Modified", "recorded_error: ''")
    with open(new_filename, "w") as f:
        f.write(content)

    print(f"New release note saved as {new_filename}")

    print(
        "\nWARNING\n"
        "=======================\n"
        "You have made changes to the output of Retuve. "
        "Please ensure these changes are intended \nand either minor, "
        "or making meaningful clinical improvements."
    )

    print(
        "\nSome values from the test-data need to be sey manually. "
        "If tests are failing, please check expected_us_metrics "
        "in ./tests/fixtures/unit.py"
    )
else:
    print("No changes detected; no new release note created.")


# print warning to inspect the results manually, specifically
# the video and xray image
print(
    "\nWARNING\n"
    "=======================\n"
    "Please inspect the results manually, specifically the video and xray image."
)
# print the file locations
print(f"Video: {test_data_dir}/video_us.mp4")
print(f"Xray Image: {test_data_dir}/img_xray.jpg")
