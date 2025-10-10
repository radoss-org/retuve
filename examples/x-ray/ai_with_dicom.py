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

import pydicom
from retuve_yolo_plugin.xray import yolo_predict_dcm_xray

from retuve.defaults.hip_configs import default_xray
from retuve.funcs import analyse_hip_xray_2D
from retuve.testdata import Cases, download_case

# dcm_file = "path/to/file"
dcm_file = download_case(Cases.XRAY_DICOM)[0]

default_xray.device = "cpu"
dcm = pydicom.dcmread(dcm_file)
hip_data, img, dev_metrics = analyse_hip_xray_2D(
    dcm,
    keyphrase=default_xray,
    modes_func=yolo_predict_dcm_xray,
    modes_func_kwargs_dict={},
)

img.save("xray.jpg")

metrics = hip_data.json_dump(default_xray, dev_metrics)
print(metrics)
print(f"Landmarks: {hip_data.landmarks}")
