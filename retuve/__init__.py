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

from importlib.metadata import version

"""
# The first fully Open Source Infant Hip Analysis Package

Retuve (from the scottish gaelic `Ri taobh` meaning `beside`) is a framework for analysing infant hips. It is designed to be a flexible and extensible framework that can be used by developers, AI researchers and clinicians.

It takes in raw Hip Ultrasound and X-Ray images, and outputs a report with the labelled images, and the results, exactly as a clinician would.

<img src="https://raw.githubusercontent.com/radoss-org/radoss-creative-commons/main/other/224_ddh_115_%26_172535_0_diagram.jpg" alt="drawing" width="500"/>

Attribution of the above Ultrasound Images: Case courtesy of Ryan Thibodeau from https://radiopaedia.org 172535 (https://radiopaedia.org/cases/172535)

Attribution of the above X-Ray Images: Fraiwan, Mohammad; Al-Kofahi, Noran; Hanatleh, Omar; ibnian, ali (2022), “A dataset of DDH x-ray images”, Mendeley Data, V2, doi: 10.17632/jf3pv98m9g.2

# Quickstart

To get started with Retuve, you can install it via pip:

```bash
pip install git+https://github.com/radoss-org/retuve.git
```

You can then run the following code to get a basic report:

```python
import pydicom
from radstract.data.dicom import convert_dicom_to_images
from retuve.defaults.hip_configs import default_US
from retuve.defaults.manual_seg import manual_predict_us
from retuve.funcs import analyse_hip_2DUS
from retuve.testdata import Cases, download_case

# Example usage
dcm_file, seg_file = download_case(Cases.ULTRASOUND_DICOM)

dcm = pydicom.dcmread(dcm_file)
images = convert_dicom_to_images(dcm)

hip_data, img, dev_metrics = analyse_hip_2DUS(
    images[0],
    keyphrase=default_US,
    modes_func=manual_predict_us,
    modes_func_kwargs_dict={"seg": seg_file},
)

img.save("2dus.png")
```
<img src="https://raw.githubusercontent.com/radoss-org/radoss-creative-commons/main/other/ultrasound/172535_0_processed.png" alt="drawing" width="500"/>

Attribution of the above Ultrasound Images: Case courtesy of Ryan Thibodeau from https://radiopaedia.org 172535 (https://radiopaedia.org/cases/172535)
# Features

- pip installable (easy to intergrate with you existing systems)
- Apache 2.0 Licensed
- AI is fully pluggable/modular
- Basic Web Interface bundled
- CLI Interface
- Swagger API Provided

# Examples

Examples can be found at https://github.com/radoss-org/retuve/tree/main/examples

# Docs

We provide high level overviews for different types of users. This includes a tailored description of Retuve, and some highlighted features:

- For Developers: `retuve.docs.overviews.developers`
- For AI Researchers: `retuve.docs.overviews.ai_researchers`
- For Clinicians: `retuve.docs.overviews.clinicians`

# Modalities

Retuve can analyse Hips for:

- Ultrasound: `retuve.hip_us`
- X-Ray: `retuve.hip_xray`

# Developer Guide

You can clone the repository and install the dependencies with the following command:

```bash
git clone https://github.com/radoss-org/retuve.git
```

You can then install retuve with poetry, and then run the tests:

```bash
# Needed for the scripts
pip install poethepoet

cd retuve
poetry install

# Generate the test data
poe testgen

# Run all tests, including examples.
poe test_all

# Get info on all other dev scripts
poe help
```

"""

__version__ = version("retuve")


def print_disclaimer():
    message = """
**Disclaimer:**

This software is an experimental tool for **Research Use Only**.
It is intended solely for research purposes related to the diagnosis of
Developmental Dysplasia of the Hip (DDH).

This software has **NOT** been approved as a medical device by any
regulatory agency in any country. It is not intended for clinical use,
diagnosis, or treatment.

The software is provided "as is" without any warranty, express or implied.
Use of this software is at your own risk. The developers and contributors are
not responsible for any consequences resulting from the use or misuse of this
software.

Please consult with qualified healthcare professionals for any medical advice
or diagnosis.
"""
    print(message)


print_disclaimer()
