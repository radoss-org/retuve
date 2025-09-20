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
Handles 3D and 2D Sweep Hip Ultrasounds that do not have the
Anterior and Posterior sides correctly set.

Retuve is set up to have the Anterior side first and the Posterior side second.
"""

from typing import List, Tuple

from retuve.classes.seg import SegFrameObjects
from retuve.hip_us.classes.enums import HipLabelsUS, Side
from retuve.hip_us.classes.general import HipDatasUS, HipDataUS
from retuve.utils import warning_decorator


def get_side_metainfo(
    hip: HipDataUS, results: List[SegFrameObjects]
) -> Tuple[Tuple[int], List[int]]:
    """
    Get the side metainfo for a HipDataUS object.

    :param hip: HipDataUS object.
    :param results: List of SegFrameObjects.
    """
    if not hip.landmarks:
        return None, None

    left = hip.landmarks.left
    apex = hip.landmarks.apex

    mid = (left[0] + apex[0]) / 2, (left[1] + apex[1]) / 2

    illium = [
        seg_obj for seg_obj in results if seg_obj.cls == HipLabelsUS.IlliumAndAcetabulum
    ]

    if len(illium) == 0:
        return None, None

    illium = illium[0]

    illium_midline = illium.midline

    # reverse x and y in midline
    illium_midline = [(y, x) for x, y in illium_midline]

    closest_illium = min(
        illium_midline,
        key=lambda point: abs(point[0] - mid[0]),
    )

    return closest_illium, mid


def set_side_info(
    hip_datas: HipDatasUS,
    results: List[SegFrameObjects],
    config,
) -> Tuple[HipDatasUS, List[SegFrameObjects]]:
    if config.hip.allow_flipping:
        return reverse_3dus_orientaition(hip_datas, results, config.hip.allow_flipping)
    for hip_data in hip_datas:
        if hip_data.frame_no < hip_datas.graf_frame:
            hip_data.side = Side.ANT
        else:
            hip_data.side = Side.POST

    return hip_datas, results


@warning_decorator(alpha=True)
def reverse_3dus_orientaition(
    hip_datas: HipDatasUS,
    results: List[SegFrameObjects],
    allow_flipping,
) -> Tuple[HipDatasUS, List[SegFrameObjects]]:
    """
    Returns the 3DUS HipDatasUS object and the results with the Anterior side first.

    :param hip_datas: HipDatasUS object.
    :param results: List of SegFrameObjects.
    :param allow_flipping: Boolean indicating if flipping is allowed.

    :return: Tuple of HipDatasUS object and List of SegFrameObjects.
    """

    # split the hip datas into two sides, based on graf frame
    front_side = []
    back_side = []

    for hip_data, seg_frame_objs in zip(hip_datas, results):
        if hip_data.landmarks is None or hip_data.landmarks.apex is None:
            continue

        closest_illium, mid = get_side_metainfo(hip_data, seg_frame_objs)

        if closest_illium is None:
            continue

        # Find the y distance between the midline points
        delta = mid[1] - closest_illium[1]

        if hip_data.frame_no < hip_datas.graf_frame:
            front_side.append(delta)
        else:
            back_side.append(delta)

    for side in [("Front", front_side), ("Back", back_side)]:  # noqa
        if len(side[1]) == 0:
            hip_datas.recorded_error.append(f"No {side[0]} Side.")
            hip_datas.recorded_error.critical = True

    if len(front_side) == 0:
        front_side = [0]
    if len(back_side) == 0:
        back_side = [0]

    if front_side == [0] and back_side == [0]:
        hip_datas.recorded_error.append("No Side Detected.")
        return hip_datas, results

    back_side_delta = sum(back_side) / len(back_side)
    front_side_delta = sum(front_side) / len(front_side)

    flip_frames = False
    if (back_side_delta > front_side_delta) and allow_flipping:
        hip_datas.recorded_error.append("Swapped Post and Ant")
        hip_datas.hip_datas = hip_datas.hip_datas[::-1]
        results = results[::-1]
        hip_datas.graf_frame = len(hip_datas) - hip_datas.graf_frame
        flip_frames = True

    # set the side for each hip data
    for i, hip_data in enumerate(hip_datas):
        if flip_frames:
            hip_data.frame_no = i
        if hip_data.frame_no < hip_datas.graf_frame:
            hip_data.side = Side.ANT
        else:
            hip_data.side = Side.POST

    return hip_datas, results
