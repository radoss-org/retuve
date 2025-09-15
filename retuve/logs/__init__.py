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
Logging utilities for the project.

Long term, this should be a decorator.
"""

import logging

MS_CONVERSION_FACTOR = 1000

ulogger = logging.getLogger("retuve")
ulogger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
ulogger.addHandler(handler)
ulogger.propagate = False


def log_timings(timings, title=None):
    if title is None:
        title = "Speed:"

    if not timings:
        ulogger.info(f"{title} No timings to log.")
        return

    total_time_ms = sum(timings)
    average_time_ms = total_time_ms / len(timings)
    ulogger.info(
        f"{title} {total_time_ms:.2f} s total, "
        f"{average_time_ms*MS_CONVERSION_FACTOR :.2f} ms per frame. "
        f"Max: {max(timings)*MS_CONVERSION_FACTOR:.2f} ms, "
    )
