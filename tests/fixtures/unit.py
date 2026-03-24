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

import pytest

from retuve.hip_us.classes.general import HipDatasUS


@pytest.fixture
def expected_xray_metrics(metrics_xray):

    metrics = {}
    for _dict in metrics_xray["metrics"]:
        metrics.update(_dict)

    return {
        "ace_left": metrics["ace_index_left"],
        "ace_right": metrics["ace_index_right"],
        "wiberg_left": metrics["wiberg_index_left"],
        "wiberg_right": metrics["wiberg_index_right"],
        "ihdi_left": metrics["ihdi_grade_left"],
        "ihdi_right": metrics["ihdi_grade_right"],
        "tonnis_grade_left": metrics["tonnis_grade_left"],
        "tonnis_grade_right": metrics["tonnis_grade_right"],
    }


@pytest.fixture
def expected_us_metrics(metrics_3d_us, metrics_2d_us):

    # metrics_3d_us["metrics"] is wrapped in lists for each key
    # i.e metrics_3d_us["metrics"][0]["Cen. ratio"], metrics_3d_us["metrics"][1]["Aca"]
    # please undo this wrapping before using the metrics
    metrics_3dus = {}
    for _dict in metrics_3d_us["metrics"]:
        metrics_3dus.update(_dict)

    # same for metrics_2d_us, but it may contain Metric2D string reprs
    metrics_2dus = {}
    for item in metrics_2d_us["metrics"]:
        if isinstance(item, dict):
            metrics_2dus.update(item)
        elif isinstance(item, str):
            # Expect format like: "Metric2D(name=alpha, value=69.2)"
            try:
                content = item.strip()
                if content.startswith("Metric2D(") and content.endswith(")"):
                    inner = content[len("Metric2D(") : -1]
                    parts = [p.strip() for p in inner.split(",")]
                    kv = {}
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            kv[k.strip()] = v.strip()
                    name = kv.get("name")
                    value = kv.get("value")
                    if name is not None and value is not None:
                        # cast numeric if possible
                        try:
                            num_val = float(value)
                        except ValueError:
                            num_val = value
                        metrics_2dus[name] = num_val
            except Exception:
                # ignore unparsable entries
                pass

    return {
        "centering_ratio": metrics_3dus["Cen. ratio"][3],
        "curvature": metrics_2dus.get("curvature", metrics_2dus.get("name=curvature")),
        "alpha": metrics_2dus.get("alpha", metrics_2dus.get("name=alpha")),
        "aca_post": metrics_3dus["Aca"][0],
        "aca_ant": metrics_3dus["Aca"][2],
        "aca_graf": metrics_3dus["Aca"][1],
        # Hack since we don't run the thirds pipeline
        "aca_thirds_range": range(55, 70),
        "outliers": [
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ],
        "frame_with_results": 4,
        "frame_with_no_results": 0,
        "flipped": False,
    }


@pytest.fixture
def edited_hips(pre_edited_hip_datas_us: HipDatasUS):
    hip_datas_us = copy.deepcopy(pre_edited_hip_datas_us)

    return hip_datas_us
