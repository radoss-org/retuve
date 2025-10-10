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
from retuve_yolo_plugin.ultrasound import yolo_predict_us

from retuve.defaults.hip_configs import default_US
from retuve.funcs import analyse_hip_2DUS
from retuve.testdata import Cases, download_case

# img_file = "path/to/file"
img_file = download_case(Cases.ULTRASOUND_JPG)[0]

img_raw = Image.open(img_file)

default_US.device = "cpu"
hip_data, img, dev_metrics = analyse_hip_2DUS(
    img_raw,
    keyphrase=default_US,
    modes_func=yolo_predict_us,
    modes_func_kwargs_dict={},
)

img_raw.save("2dus-raw.png")
img.save("2dus.png")

metrics = hip_data.json_dump(default_US, dev_metrics)
print(metrics)
