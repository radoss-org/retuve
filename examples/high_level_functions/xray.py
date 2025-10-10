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

import json

from PIL import Image

from retuve.defaults.hip_configs import default_xray
from retuve.defaults.manual_seg import manual_predict_xray
from retuve.funcs import analyse_hip_xray_2D
from retuve.testdata import Cases, download_case

# Example usage
# dcm_file = "path/to/file"
# seg_file = "/path/to/nifti"
img_file, labels_json = download_case(Cases.XRAY_JPG)

img_raw = Image.open(img_file)
labels = json.load(open(labels_json))

hip_data, img, dev_metrics = analyse_hip_xray_2D(
    img_raw,
    keyphrase=default_xray,
    modes_func=manual_predict_xray,
    modes_func_kwargs_dict=labels,
)

img.save("xray.jpg")
img_raw.save("xray-raw.jpg")

metrics = hip_data.json_dump(default_xray, dev_metrics)
print(metrics)
