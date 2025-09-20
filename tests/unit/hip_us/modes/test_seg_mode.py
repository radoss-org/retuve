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

import numpy as np

from retuve.hip_us.classes.general import LandmarksUS
from retuve.hip_us.modes.seg import (
    get_midlines,
    pre_process_segs_us,
    segs_2_landmarks_us,
)


def test_get_midlines(illium_0, config_us):
    old_illium = copy.deepcopy(illium_0)
    illium_0.midline = None
    illium_0.midline_moved = None

    midline, midline_moved = get_midlines(illium_0.mask, config_us)
    assert midline is not None
    assert midline_moved is not None
    assert np.array_equal(old_illium.midline, midline)
    assert np.array_equal(old_illium.midline_moved, midline_moved)


def test_segs_2_landmarks_us(pre_edited_landmarks_us, pre_edited_results_us, config_us):
    new_landmarks_list, *_ = segs_2_landmarks_us(pre_edited_results_us, config_us)

    # remove landmarks that are None
    pre_edited_landmarks_us = [
        landmark for landmark in pre_edited_landmarks_us if landmark is not None
    ]
    new_landmarks_list = [
        landmarks for landmarks in new_landmarks_list if landmarks is not None
    ]

    # assert their lengths are the same
    assert len(pre_edited_landmarks_us) == len(new_landmarks_list)
    for old_landmarks, new_landmarks in zip(
        pre_edited_landmarks_us, new_landmarks_list
    ):
        assert isinstance(new_landmarks, LandmarksUS)
        assert old_landmarks.left == new_landmarks.left
        assert old_landmarks.right == new_landmarks.right
        assert old_landmarks.apex == new_landmarks.apex
        assert old_landmarks.point_D == new_landmarks.point_D
        assert old_landmarks.point_d == new_landmarks.point_d


def test_pre_process_segs_us(results_us, config_us):
    processed_results, shape = pre_process_segs_us(results_us, config_us)
    assert len(processed_results) == len(results_us)
    assert shape == results_us[0].img.shape
