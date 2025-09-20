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
All code relating to going from measured landmarks to metrics is in this file.
"""

import time
from typing import List, Tuple

import numpy as np

from retuve.classes.metrics import Metric2D
from retuve.hip_us.classes.general import HipDatasUS, HipDataUS, LandmarksUS
from retuve.hip_us.metrics.alpha import find_alpha_angle
from retuve.hip_us.metrics.coverage import find_coverage
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import MetricUS
from retuve.logs import log_timings


def _polyfit_replace_apex(
    list_landmarks: List[LandmarksUS],
    degree: int = 2,
    max_pixel_err: float = 8.0,
) -> None:
    """
    Replace apex outliers using a simple polyfit across frames (time as z).

    Fits a polynomial separately for x(t) and y(t) over frames with valid apex,
    then replaces any apex whose distance to the fit exceeds max_pixel_err.

    This function mutates list_landmarks in-place and keeps things simple:
    - ignores frames with missing apex
    - does not attempt to in-fill missing apex
    - guards for small sample sizes
    """
    # Collect indices and apex coords where available
    idxs = []
    xs = []
    ys = []
    for i, lm in enumerate(list_landmarks):
        if lm and lm.apex is not None:
            x, y = lm.apex
            if x is None or y is None:
                continue
            idxs.append(i)
            xs.append(float(x))
            ys.append(float(y))

    # Need enough points to fit the requested degree
    if len(idxs) < max(degree + 1, 3):
        return

    t = np.array(idxs, dtype=float)
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    # Fit polynomials x(t), y(t)
    try:
        px = np.poly1d(np.polyfit(t, x_arr, deg=degree))
        py = np.poly1d(np.polyfit(t, y_arr, deg=degree))
    except Exception:
        # Be conservative: if fit fails, do nothing
        return

    # Replace outliers
    for i, x_true, y_true in zip(idxs, x_arr, y_arr):
        x_fit = float(px(i))
        y_fit = float(py(i))
        # Euclidean residual in pixels
        resid = ((x_true - x_fit) ** 2 + (y_true - y_fit) ** 2) ** 0.5
        if resid > max_pixel_err:
            # Replace with rounded ints to match tuple[int, int]
            lm = list_landmarks[i]
            list_landmarks[i] = LandmarksUS(
                left=lm.left,
                right=lm.right,
                apex=(int(round(x_fit)), int(round(y_fit))),
                point_D=lm.point_D,
                point_d=lm.point_d,
                mid_cov_point=lm.mid_cov_point,
            )


def landmarks_2_metrics_us(
    list_landmarks: List[LandmarksUS],
    shape: Tuple[int, int],
    config: Config,
) -> HipDatasUS:
    """
    Converts a list of landmarks to a list of metrics.

    :param list_landmarks: A list of landmarks.
    :param shape: The shape of the image.
    :param config: The configuration object.

    :return: A list of metrics.
    """
    hips = HipDatasUS()
    timings = []

    if config.hip.use_polyfit_replace_apex:
        _polyfit_replace_apex(list_landmarks)

    for frame_no, landmarks in enumerate(list_landmarks):
        start = time.time()

        alpha = find_alpha_angle(landmarks)
        coverage = find_coverage(landmarks)

        metrics = []

        for name, value in [
            (MetricUS.ALPHA, alpha),
            (MetricUS.COVERAGE, coverage),
        ]:
            if name in config.hip.measurements:
                metrics.append(Metric2D(name, value))

        hip = HipDataUS(
            landmarks=landmarks,
            metrics=metrics,
            frame_no=frame_no,
        )

        hips.append(hip)
        timings.append(time.time() - start)

    log_timings(timings, title="Landmarks->Metrics Speed:")

    return hips
