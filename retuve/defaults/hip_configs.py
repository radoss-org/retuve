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
Default Working Configs for Retuve

Import them directly from the retuve package.

We don't store them, so that the Retuve UI does not use them.
"""

from radstract.data.dicom import DicomTypes

from retuve.keyphrases.enums import MetricUS, OperationType

from .config import base_config

default_US = base_config.get_copy()
default_US.visuals.default_font_size = 20
default_US.visuals.points_radius = 10
default_US.visuals.line_thickness = 3
default_US.visuals.bounding_box_thickness = 8
default_US.hip.display_side = False
default_US.visuals.display_segs = True
default_US.batch.input_types = [".dcm", ".jpg", ".png"]

default_US.register(name="default_US", store=False, silent=True)

default_xray = base_config.get_copy()
default_xray.operation_type = OperationType.LANDMARK
default_xray.visuals.points_radius = 10
default_xray.visuals.line_thickness = 3
default_xray.visuals.default_font_size = 20
default_xray.visuals.bounding_box_thickness = 7
default_xray.dicom_type = DicomTypes.SINGLE
default_xray.batch.input_types = [".dcm", ".jpg", ".png"]

default_xray.register(name="default_xray", store=False, silent=True)

live = default_US.get_copy()

test_default_US = default_US.get_copy()
test_default_US.hip.measurements = [
    MetricUS.ALPHA,
    MetricUS.COVERAGE,
    MetricUS.CENTERING_RATIO,
    MetricUS.ACA,
]
