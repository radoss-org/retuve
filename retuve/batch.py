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
This module contains the functions to run the retuve pipeline on a batch of
files. It also has the functions used to make the CLI commands for running retuve
on a single file or a batch of files.
"""

import json
import multiprocessing
import os
import shutil
import time
import traceback

from retuve.funcs import retuve_run
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Outputs
from retuve.logs import ulogger


def run_single(
    config: Config,
    file_name: str,
    for_batch: bool = False,
    local_savedir: str = None,
):
    """
    Run the retuve pipeline on a single file.

    :param config: The configuration.
    :param file_name: The file name.
    :param for_batch: Whether the file is being processed in a batch.
                      This changes the way the file is saved.
    :param local_savedir: The local save directory.

    :return: The error message if any.
    """

    fileid = file_name.split("/")[-1].split(".")[0]
    if local_savedir:
        savedir = local_savedir
    else:
        savedir = config.api.savedir

    hip_mode = config.batch.hip_mode

    if for_batch:
        fileid += "/"

        # if the metrics.json already exist, skip
        if os.path.exists(f"{savedir}/{fileid}metrics.json"):
            return f"File {fileid} already processed"

        if os.path.exists(f"{savedir}/{fileid}"):
            shutil.rmtree(f"{savedir}/{fileid}")

        os.makedirs(f"{savedir}/{fileid}", exist_ok=True)
    else:
        fileid += "_"

    try:
        retuve_result = retuve_run(
            hip_mode=hip_mode,
            config=config,
            modes_func=config.batch.mode_func,
            modes_func_kwargs_dict=config.batch.mode_func_args,
            file=file_name,
        )
        hip_datas = retuve_result.hip_datas

        if hip_datas and hip_datas.recorded_error:
            ulogger.info(
                f"\n Recorded Error: {hip_datas.recorded_error} "
                f"Critical: {hip_datas.recorded_error.critical}"
            )

        if retuve_result.image is not None:
            retuve_result.image.save(f"{savedir}/{fileid}{Outputs.IMAGE}")

        if retuve_result.metrics and retuve_result.metrics.get("dev_metrics"):
            ulogger.info(
                "\n Dev Metrics: ", retuve_result.metrics["dev_metrics"]
            )

        if retuve_result.video_clip is not None:
            retuve_result.video_clip.write_videofile(
                f"{savedir}/{fileid}{Outputs.VIDEO_CLIP}",
            )

        if retuve_result.visual_3d is not None:
            retuve_result.visual_3d.write_html(
                f"{savedir}/{fileid}{Outputs.VISUAL3D}"
            )

        if config.seg_export and hip_datas and hip_datas.nifti is not None:
            hip_datas.nifti.save(f"{savedir}/{fileid}{Outputs.NIFTI}")

        # save the metrics to a file
        with open(f"{savedir}/{fileid}{Outputs.METRICS}", "w") as f:
            f.write(json.dumps(retuve_result.metrics))

    except Exception as e:
        e = traceback.format_exc()
        ulogger.error(f"Error processing file {file_name}: {e}")
        return e


def run_batch(config: Config):
    """
    Run the retuve pipeline on a batch of files.

    :param config: The configuration.
    """
    all_files = []

    for dataset in config.batch.datasets:
        files = os.listdir(dataset)

        files = [
            f"{dataset}/{file}"
            for file in files
            if any(
                file.endswith(input_type)
                for input_type in config.batch.input_types
            )
        ]

        all_files.extend(files)

    start = time.time()

    # create savedir if it doesn't exist
    if not os.path.exists(config.api.savedir):
        os.makedirs(config.api.savedir, exist_ok=True)

    if not multiprocessing.get_start_method(allow_none=True):
        multiprocessing.set_start_method("spawn", force=True)

    with multiprocessing.Pool(processes=config.batch.processes) as pool:
        chunks = [(config, file, True) for file in all_files]
        errors = pool.starmap(run_single, chunks)

    if any(error is not None for error in errors):
        already_processed = sum(
            "already processed" in error
            for error in errors
            if error is not None
        )
        # count and remove all errors containing "already processed"
        errors = [
            error
            for error in errors
            if error is not None and "already processed" not in error
        ]

        for error in errors:
            print(error)

        print(f"Errors: {len(errors)}")
        print(f"Already processed: {already_processed}")

    end = time.time()

    if len(all_files) == 0:
        print(
            f"No files with types in {config.batch.input_types} "
            "found in the directory"
        )
        return

    # convert to minutes and seconds
    minutes, seconds = divmod(end - start, 60)
    print(f"Time taken: {minutes:.0f}m {seconds:.0f}s")

    # Half to ignore the .nii files
    no_of_files = len(all_files)

    if no_of_files == 0:
        no_of_files = 1

    # Print average time per file
    print(f"Average time per file: {(end - start) / no_of_files:.2f}s")

    # Print number of files
    print(f"Number of files: {no_of_files}")
