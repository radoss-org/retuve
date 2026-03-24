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

# Define the Apache License text
LICENSE_NOTICE = """# Copyright 2024 Adam McArthur
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


def add_license_to_file(filepath):
    """Insert the license notice at the top of the file if not already present."""
    with open(filepath, "r") as file:
        content = file.read()

    if LICENSE_NOTICE.strip() in content:
        print(f"License already present in {filepath}. Skipping.")
        return

    with open(filepath, "w") as file:
        file.write(LICENSE_NOTICE + "\n" + content)
        print(f"License added to {filepath}.")


def process_directory(directory):
    """Process all .py files in the given directory recursively."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                add_license_to_file(filepath)


if __name__ == "__main__":
    target_directories = ["./retuve", "./examples", "./tests"]

    for target_directory in target_directories:
        if os.path.isdir(target_directory):
            process_directory(target_directory)
            print("Processing completed.")
        else:
            print(f"Invalid directory: {target_directory}")
