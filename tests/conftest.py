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
import gzip
import json
import logging
import os
import pickle
from typing import List

import afterthought
import numpy as np
import open3d as o3d
import pytest
import yaml
from PIL import Image, ImageOps

from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.defaults.hip_configs import (
    default_US,
    default_xray,
    test_default_US,
)
from retuve.draw import TARGET_SIZE
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.hip_us.classes.general import HipDatasUS, LandmarksUS
from retuve.hip_xray.classes import HipDataXray, LandmarksXRay
from retuve.keyphrases.config import Config

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


default_US.register("ultrasound", live=True)


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["RETUVE_DISABLE_WARNING"] = "true"


# check if test-data exists, otherwise print a message
if not os.path.exists("tests/test-data"):
    logging.error("No test data found. Please run `poe testgen`")
    exit(1)

with gzip.open("tests/test-data/hip_datas_us.pkl.gz", "rb") as f:
    data_hip_datas: HipDatasUS = pickle.load(f)

with gzip.open("tests/test-data/results_us.pkl.gz", "rb") as f:
    data_results: List[SegFrameObjects] = pickle.load(f)

# Open the mesh
data_illium_mesh = o3d.io.read_triangle_mesh("tests/test-data/illium_mesh.obj")

# Open Fem_sph
with gzip.open("tests/test-data/fem_sph.pkl.gz", "rb") as f:
    data_fem_sph = pickle.load(f)

# Open apex_points
with gzip.open("tests/test-data/apex_points.pkl.gz", "rb") as f:
    data_apex_points = pickle.load(f)

with gzip.open("./tests/test-data/femoral_sphere.pkl.gz", "rb") as f:
    data_femoral_sphere = pickle.load(f)

# avg_normals_data and normals_data
with gzip.open("./tests/test-data/avg_normals_data.pkl.gz", "rb") as f:
    data_avg_normals_data = pickle.load(f)

with gzip.open("./tests/test-data/normals_data.pkl.gz", "rb") as f:
    data_normals_data = pickle.load(f)

with gzip.open("./tests/test-data/pre_edited_hip_datas.pkl.gz", "rb") as f:
    data_pre_edited_hip_datas = pickle.load(f)

with gzip.open("./tests/test-data/pre_edited_results.pkl.gz", "rb") as f:
    data_pre_edited_results = pickle.load(f)


with gzip.open("./tests/test-data/pre_edited_landmarks.pkl.gz", "rb") as f:
    data_pre_edited_landmarks = pickle.load(f)


with open(f"tests/test-data/331_DDH_115.json", "r") as f:
    json_xray_landmarks = f.read()


@pytest.fixture
def pre_edited_landmarks_us() -> List[LandmarksUS]:
    return copy.deepcopy(data_pre_edited_landmarks)


@pytest.fixture
def pre_edited_hip_datas_us() -> HipDatasUS:
    return copy.deepcopy(data_pre_edited_hip_datas)


@pytest.fixture
def pre_edited_results_us() -> List[SegFrameObjects]:
    return copy.deepcopy(data_pre_edited_results)


@pytest.fixture
def hip_datas_us() -> HipDatasUS:
    return copy.deepcopy(data_hip_datas)


@pytest.fixture
def results_us() -> List[SegFrameObjects]:
    return copy.deepcopy(data_results)


@pytest.fixture
def illium_mesh() -> o3d.geometry.TriangleMesh:
    return copy.deepcopy(data_illium_mesh)


@pytest.fixture
def fem_sph() -> SegObject:
    return copy.deepcopy(data_fem_sph)


@pytest.fixture
def apex_points() -> List:
    return copy.deepcopy(data_apex_points)


@pytest.fixture
def femoral_sphere() -> SegObject:
    return copy.deepcopy(data_femoral_sphere)


@pytest.fixture
def landmarks_us() -> List[LandmarksUS]:
    return [
        copy.deepcopy(data_hip_datas[i].landmarks) for i in range(len(data_hip_datas))
    ]


@pytest.fixture
def hip_data_us_0(expected_us_metrics) -> HipDatasUS:
    return copy.deepcopy(data_hip_datas[expected_us_metrics["frame_with_results"]])


@pytest.fixture
def results_us_0(expected_us_metrics) -> SegFrameObjects:
    return copy.deepcopy(data_results[expected_us_metrics["frame_with_results"]])


@pytest.fixture
def illium_0(expected_us_metrics) -> SegObject:
    illium_obj = [
        seg_obj
        for seg_obj in data_results[expected_us_metrics["frame_with_results"]]
        if seg_obj.cls == HipLabelsUS.IlliumAndAcetabulum
    ][0]

    return copy.deepcopy(illium_obj)


@pytest.fixture
def femoral_0(expected_us_metrics) -> SegObject:
    femoral_obj = [
        seg_obj
        for seg_obj in data_results[expected_us_metrics["frame_with_results"]]
        if seg_obj.cls == HipLabelsUS.FemoralHead
    ][0]

    return copy.deepcopy(femoral_obj)


@pytest.fixture
def landmarks_us_0(expected_us_metrics) -> LandmarksUS:
    return copy.deepcopy(
        data_hip_datas[expected_us_metrics["frame_with_results"]]
    ).landmarks


@pytest.fixture
def hip_data_xray_0() -> HipDataXray:
    return copy.deepcopy(data_hip_data_xray_0)


@pytest.fixture
def landmarks_xray_0(hip_data_xray_0) -> LandmarksXRay:
    return hip_data_xray_0.landmarks


@pytest.fixture
def avg_normals_data() -> List:
    return copy.deepcopy(data_avg_normals_data)


@pytest.fixture
def normals_data() -> List:
    return copy.deepcopy(data_normals_data)


@pytest.fixture
def img_shape_us(results_us_0) -> tuple:
    empty_img = np.zeros(results_us_0.img.shape, dtype=np.uint8)

    return ImageOps.contain(Image.fromarray(empty_img), (TARGET_SIZE)).size[:2]


@pytest.fixture
def config_us() -> Config:
    return test_default_US


@pytest.fixture
def config_xray() -> Config:
    return default_xray


# -----------------------------------------------------------------------------
# Fixtures: custom-drawn images generated by tests/_get_test_data.py
# -----------------------------------------------------------------------------


@pytest.fixture
def img_2dus_custom_np() -> np.ndarray:
    """Load the 2DUS image with custom post-draw annot (numpy array)."""
    path = "tests/test-data/img_2dus_custom.jpg"
    assert os.path.exists(path), "Custom 2DUS image not found. Run `poe testgen`."
    return np.array(Image.open(path))


@pytest.fixture
def img_xray_custom_np() -> np.ndarray:
    """Load the X-ray image with custom post-draw annot (numpy array)."""
    path = "tests/test-data/img_xray_custom.jpg"
    assert os.path.exists(path), "Custom X-ray image not found. Run `poe testgen`."
    return np.array(Image.open(path))


@pytest.fixture
def img_3dus_custom_np() -> np.ndarray:
    """Load the 3DUS Graf-frame image with custom post-draw annot (numpy array)."""
    path = "tests/test-data/img_3dus_custom.jpg"
    assert os.path.exists(path), "Custom 3DUS image not found. Run `poe testgen`."
    return np.array(Image.open(path))


def pytest_addoption(parser):
    parser.addoption(
        "--capture-errors",
        action="store_true",
        default=False,
        help="Capture and pass errors to afterthought",
    )


# This will hold all the exceptions
exceptions = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call" and result.failed:
        error = {
            "nodeid": result.nodeid,
            "location": result.location,
            "longrepr": str(result.longrepr),
            "exception": call.excinfo.value,  # Capture the exception object
        }
        exceptions.append(error)


def pytest_sessionfinish(session, exitstatus):
    capture_errors = session.config.getoption("--capture-errors")

    if not capture_errors:
        return

    # This hook is called after the test session ends
    for error in exceptions:
        afterthought.debug(error=error["exception"])


# Save the data
with gzip.open("./tests/test-data/hip_data_xray.pkl.gz", "rb") as f:
    data_hip_data_xray_0 = pickle.load(f)

with gzip.open("./tests/test-data/seg_results_xray.pkl.gz", "rb") as f:
    data_seg_results_xray = pickle.load(f)


@pytest.fixture
def seg_results_xray() -> List[SegFrameObjects]:
    return copy.deepcopy(data_seg_results_xray)


@pytest.fixture
def img_shape_xray(seg_results_xray) -> tuple:
    return seg_results_xray[0].img.shape


@pytest.fixture
def landmarks_xray():
    return json.loads(json_xray_landmarks)


pytest_plugins = [
    "tests.fixtures.intergration",
    "tests.fixtures.unit",
]


def load_previous_release_note():
    # Locate the most recent YAML file in ./changenotes/
    release_dir = "./changenotes/"
    yaml_files = [f for f in os.listdir(release_dir) if f.endswith(".yaml")]
    yaml_files.sort(reverse=True)
    if yaml_files:
        with open(os.path.join(release_dir, yaml_files[0]), "r") as f:
            return yaml.safe_load(f), yaml_files[0]
    return None, None


previous_data, previous_filename = load_previous_release_note()
current_output = previous_data["current_output"]


@pytest.fixture
def metrics_3d_us():
    return current_output["3dus"]


@pytest.fixture
def metrics_xray():
    return current_output["xray"]


@pytest.fixture
def metrics_2d_us():
    return current_output["2dus"]


@pytest.fixture
def metrics_2d_sweep():
    return current_output["us_sweep"]


@pytest.fixture
def ihdi_points_left():
    return [
        (3, 3),
        (4, 6),
        (8, 6),
        ((4, 6), (0, 10)),
    ]


@pytest.fixture
def ihdi_points_right():
    return [
        (9, 3),
        (4, 6),
        (8, 6),
        ((8, 6), (12, 10)),
    ]
