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

from retuve.defaults.hip_configs import default_US
from retuve.defaults.manual_seg import manual_predict_us_dcm
from retuve.funcs import analyse_hip_2DUS_sweep
from retuve.testdata import Cases, download_case

# Example usage
dcm_file, seg_file = download_case(Cases.ULTRASOUND_DICOM)

dcm = pydicom.dcmread(dcm_file)

hip_datas, img, dev_metrics, video_clip = analyse_hip_2DUS_sweep(
    dcm,
    keyphrase=default_US,
    modes_func=manual_predict_us_dcm,
    modes_func_kwargs_dict={"seg": seg_file},
)

video_clip.write_videofile("2dus_sweep.mp4")
img.save("2dus_sweep.png")

metrics = hip_datas.json_dump(default_US, dev_metrics)
print(metrics)
