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
from retuve_yolo_plugin.ultrasound import yolo_predict_dcm_us

from retuve.defaults.hip_configs import default_US

# to do a simple sweep analysis,
# from retuve.funcs import analyse_hip_2DUS_sweep
# instead (examples/high_level_functions/2dus_sweep.py)
from retuve.funcs import analyse_hip_3DUS
from retuve.testdata import Cases, download_case

# dcm_file = "path/to/file"
dcm_file, _ = download_case(Cases.ULTRASOUND_DICOM)

default_US.device = "cpu"
dcm = pydicom.dcmread(dcm_file)
hip_datas, video_clip, visual_3d, dev_metrics = analyse_hip_3DUS(
    dcm,
    keyphrase=default_US,
    modes_func=yolo_predict_dcm_us,
    modes_func_kwargs_dict={},
)


video_clip.write_videofile("3dus.mp4")
visual_3d.write_html("3dus.html")

metrics = hip_datas.json_dump(default_US)
print(metrics)
