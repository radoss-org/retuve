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

from PIL import Image

# change this to
# from retuve_yolo_plugin.xray_pose import yolo_predict_xray
# to use the landmark model vs the triangle model
from retuve_yolo_plugin.xray import yolo_predict_xray

from retuve.defaults.hip_configs import default_xray
from retuve.funcs import analyse_hip_xray_2D
from retuve.testdata import Cases, download_case

# img_file = "path/to/file"
img_file, _ = download_case(Cases.XRAY_JPG)

default_xray.device = "cpu"
img_raw = Image.open(img_file)
hip_data, img, dev_metrics = analyse_hip_xray_2D(
    img_raw,
    keyphrase=default_xray,
    modes_func=yolo_predict_xray,
    modes_func_kwargs_dict={},
)

img.save("xray.jpg")
img_raw.save("xray-raw.jpg")

metrics = hip_data.json_dump(default_xray, dev_metrics)
print(metrics)
print(f"Landmarks: {hip_data.landmarks}")
