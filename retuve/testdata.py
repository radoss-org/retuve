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
Provides a convenient way to download test data for the Retuve library.
"""

import os
import sys
from enum import Enum

from radstract.testdata import download_case as download_case_radstract

URL = "https://github.com/radoss-org/radoss-creative-commons/raw/185aed296005617d13bc959b9e2853749c524586"


def download_case(*args, disable_warning=False, **kwargs):
    """
    Download a test case from the Radoss repository.
    """

    # identify if running inside pytest
    if os.environ.get("RETUVE_DISABLE_WARNING", "").lower() == "true":
        disable_warning = True

    if disable_warning:
        return download_case_radstract(*args, **kwargs)

    print(
        """
    DISCLAIMER
    =======================
    Before running any examples, please read the following disclaimer carefully:

    The data used by these examples is licensed under the CC BY-NC-SA 3.0 license, as detailed here:
    https://github.com/radoss-org/radoss-creative-commons

    By running examples with this data, you agree to abide by the terms of the CC BY-NC-SA 3.0 license:
    - You may share and adapt the data, but you must give appropriate credit, provide a link to the license, and indicate if changes were made.
    - You may not use the data for commercial purposes.
    - If you remix, transform, or build upon the data, you must distribute your contributions under the same license.

    If you do not agree to these terms, do not proceed with running these examples.

    Please type "yes" to confirm that you have read and agree to the terms of the CC BY-NC-SA 3.0 license.
    """
    )

    user_input = input(
        "Do you agree to the terms of the CC BY-NC-SA 3.0 license? (Type 'yes' to continue): "
    )

    if user_input.lower() != "yes":
        print("You did not agree to the terms. Exiting...")
        exit()

    print(
        "Thank you for agreeing to the terms. Proceeding with the test generation..."
    )

    return download_case_radstract(*args, **kwargs)


class Cases(Enum):
    """
    The test cases available for download.
    """

    ULTRASOUND_DICOM = [
        f"{URL}/dicoms/ultrasound/171551.dcm",
        f"{URL}/labels/ultrasound/171551.nii.gz",
    ]

    XRAY_JPG = [
        f"{URL}/other/xray/331_DDH_115.jpg",
        f"{URL}/labels/xray/331_DDH_115.json",
    ]
