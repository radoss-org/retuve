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
Handles Segmentaitions that are not valid for analysis.
"""

from typing import Dict, List

from PIL import Image
from retuve.classes.seg import SegObject
from retuve.hip_us.classes.enums import HipLabelsUS


def remove_bad_objs(
    hip_objs: Dict[HipLabelsUS, List[SegObject]], img: Image.Image
) -> Dict[HipLabelsUS, SegObject]:
    """
    Remove bad segmentation objects from the Hip Objects.

    For example, if there are multiple segmentations for the Illium and Acetabulum,
    only keep the one that is the most left of the image.

    :param hip_objs: Dict of Hip Objects.
    :param img: Image object.

    """

    rejection_reasons = []

    fem_head_ilium_wrong_way_round = False
    if len(hip_objs[HipLabelsUS.IlliumAndAcetabulum]) > 1:
        hip_objs[HipLabelsUS.IlliumAndAcetabulum] = [
            sorted(
                hip_objs[HipLabelsUS.IlliumAndAcetabulum],
                key=lambda seg_obj: seg_obj.box[0],
            )[0]
        ]

    # replace each value with the largest object
    for k, v in hip_objs.items():
        if len(v) != 0:
            hip_objs[k] = max(v, key=lambda seg_obj: seg_obj.area())
        else:
            hip_objs[k] = SegObject(empty=True)

    fem_head = hip_objs.get(HipLabelsUS.FemoralHead, None)

    illium = hip_objs.get(HipLabelsUS.IlliumAndAcetabulum, None)
    if illium and illium.box is not None and illium.box[0] > img.shape[1] / 2:
        # check if the femoral head box is left of the illium box
        if fem_head and fem_head.box is not None and illium.box[0] > fem_head.box[0]:
            fem_head_ilium_wrong_way_round = True

    # Femoral Heads should be at least 2.5%
    expected_min_fem_size = img.shape[0] * img.shape[1] * 0.025
    if fem_head and fem_head.area() < expected_min_fem_size:
        hip_objs[HipLabelsUS.FemoralHead] = SegObject(empty=True)
        rejection_reasons.append("Femoral Head too small")

    return hip_objs, fem_head_ilium_wrong_way_round, rejection_reasons
