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
import subprocess

import requests

to_download = [
    (
        "radoss-org",
        "retuve-yolo-plugin",
        "main",
        "tests",
        "./tests/_external_plugin/yolo",
    ),
    (
        "radoss-org",
        "retuve-tinyunet-plugin",
        "main",
        "tests",
        "./tests/_external_plugin/tinyunet",
    ),
]


for owner, repo, branch, directory, output_dir in to_download:

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}?ref={branch}"

    normal_url = f"https://github.com/{owner}/{repo}"

    # DISCLAIMER
    # =======================
    # Before running this code, please read the following disclaimer carefully:

    print(
        f"""
    DISCLAIMER
    =======================
    Before running the plugin download script, please read the following disclaimer carefully:

    The plugins have their own set of licences depending on the model used and the data used for training.

    By running this script to automatically download the plugins, you agree to abide by the terms of all their licenses.

    Please read each plugin's license(s) carefully before using them:
    - {normal_url}

    If you do not agree to these terms, do not proceed with running these tests.

    Please type "yes" to confirm that you have read and agree to the terms of all the licenses.
    """
    )

    if os.environ.get("RETUVE_DISABLE_WARNING") != "True":
        user_input = input(
            "Do you agree to the terms of the licences in all the plugins? (Type 'yes' to continue): "
        )
    else:
        user_input = "yes"

    if user_input.lower() != "yes":
        print("You did not agree to the terms. Exiting...")
        exit()

    print(
        "Thank you for agreeing to the terms. Proceeding with the test generation..."
    )

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Function to download a file
    def download_file(file_url, output_path):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {output_path}")
        else:
            print(
                f"Failed to download {file_url}. Status code: {response.status_code}"
            )

    # Fetch the list of files in the directory
    print(f"Fetching file list from {api_url}...")
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["type"] == "file":  # Only process files
                file_url = file["download_url"]
                file_name = file["name"]
                # if name starts with test_ then download it
                if not file_name.startswith("test_"):
                    continue

                name = output_dir.split("/")[-1]
                file_name = file_name.replace(".py", "")
                output_path = os.path.join(
                    output_dir, file_name + name + ".py"
                )
                download_file(file_url, output_path)
    else:
        print(
            f"Failed to fetch file list. Status code: {response.status_code}"
        )
        print(response.json())  # Print error details

    # Install the package using pip
    print("Installing the package from the repository...")
    subprocess.run(["pip", "install", "ultralytics"], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            f"git+https://github.com/{owner}/{repo}.git",
        ],
        check=True,
    )

    print(
        f"\nAll files from '{directory}' have been downloaded to '{output_dir}', and the package has been installed."
    )
