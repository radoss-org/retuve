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

import numpy as np

from retuve.classes.draw import Overlay
from retuve.hip_us.draw import draw_fem_head, draw_hips_us, draw_table


def test_draw_fem_head(hip_data_us_0, fem_sph, config_us):
    overlay = Overlay((512, 512, 3), config_us)
    z_gap = 1.0
    result_overlay = draw_fem_head(hip_data_us_0, fem_sph, overlay, z_gap)

    assert isinstance(result_overlay, Overlay)


def test_draw_hips_us(hip_datas_us, results_us, config_us, fem_sph, img_shape_us):
    images, nifti = draw_hips_us(hip_datas_us, results_us, fem_sph, config_us)

    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    # Double because we put the original image next to the overlayed image
    assert images[0].shape == (img_shape_us[1], img_shape_us[0] * 2, 3)


def test_draw_table(hip_datas_us, img_shape_us):
    image = draw_table(img_shape_us, hip_datas_us)

    assert isinstance(image, np.ndarray)
    assert image.shape == (img_shape_us[0], img_shape_us[1], 3)
