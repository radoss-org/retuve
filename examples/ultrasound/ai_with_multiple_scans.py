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

import os

from retuve_yolo_plugin.ultrasound import (
    get_yolo_model_us,
    yolo_predict_dcm_us,
)

from retuve.batch import run_batch
from retuve.defaults.hip_configs import default_US
from retuve.keyphrases.enums import HipMode
from retuve.testdata import Cases, download_case

# Or DATASET_DIR = "path/to/your/files"
dcm_file, *_ = download_case(Cases.ULTRASOUND_DCM_DATASET)
DATASET_DIR = os.path.join(os.path.dirname(dcm_file))

my_config = default_US.get_copy()
my_config.batch.hip_mode = HipMode.US3D  # or HipMode.US2DSW
my_config.batch.mode_func = yolo_predict_dcm_us
my_config.batch.mode_func_args = {"model": get_yolo_model_us(my_config)}
my_config.batch.input_types = [".dcm"]
my_config.batch.datasets = [DATASET_DIR]  # Or ["path/to/your/files"]
my_config.api.savedir = "./.test-output"
my_config.device = "cpu"


if __name__ == "__main__":
    run_batch(my_config)
