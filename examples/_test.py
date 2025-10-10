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
import shutil
import subprocess

import pytest

BLACKLIST = ["custom_ai_and_config.py"]


# Dynamically collect scripts for testing
def collect_scripts():
    examples_dir = os.path.dirname(__file__)
    module_dir = os.path.abspath(
        os.path.join(examples_dir, "..")
    )  # Assuming your module is one level up from the examples directory

    scripts = []
    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py") and file not in BLACKLIST:
                scripts.append((os.path.join(root, file), module_dir))
    return scripts


# Dynamically create test functions for pytest
scripts = collect_scripts()
for i, (script_path, module_dir) in enumerate(scripts):
    # test_name should be derived from the script name
    test_name = f'test_{script_path.split("/")[-1].replace(".py", "").replace("_", "")}'

    # divide i to fit into 4 groups
    group_no = i % 4 + 1

    def create_test(script_path, module_dir, temp_dir):
        @pytest.mark.xdist_group(name=f"group{group_no}")
        def test():
            try:
                # Capture the initial state of the directory
                initial_files = set(os.listdir(temp_dir))

                # Set up the environment
                env = os.environ.copy()
                env["PYTHONPATH"] = module_dir
                env["RETUVE_DISABLE_WARNING"] = "True"

                # Run the script
                subprocess.run(["python", script_path], check=True, env=env)

            except subprocess.CalledProcessError as e:
                pytest.fail(f"Script failed: {script_path}\nError message: {e}")

            finally:
                # Capture the state of the directory after the script runs
                final_files = set(os.listdir(temp_dir))

                # Identify new files created during script execution
                new_files = final_files - initial_files

                # Filter only files with the specified extensions
                target_extensions = {".html", ".mp4", ".jpg", ".png"}
                files_to_remove = [
                    file_name
                    for file_name in new_files
                    if os.path.splitext(file_name)[1].lower() in target_extensions
                ]

                # Delete the filtered files
                for file_name in files_to_remove:
                    file_path = os.path.join(temp_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                try:
                    test_output_dir = f"{temp_dir}/.test-output"
                    if os.path.isdir(test_output_dir):
                        shutil.rmtree(test_output_dir)
                except FileNotFoundError:
                    pass

        return test

    # Add the test function to the global scope
    globals()[test_name] = create_test(script_path, module_dir, "./")
