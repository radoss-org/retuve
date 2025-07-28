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
API routes for the Retuve App.
"""

import json
import os
import shutil

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from retuve.app.classes import FeedbackRequest, SystemFiles
from retuve.app.helpers import RESULTS_DIR
from retuve.app.routes.live import dicom_processing_queue
from retuve.app.utils import validate_api_token
from retuve.keyphrases.config import Config
from retuve.logs import ulogger
from retuve.trak.data import extract_files

router = APIRouter()


@router.post("/api/store_feedback/{keyphrase}")
async def store_feedback(
    request: Request, feedback_request: FeedbackRequest, keyphrase: str
):
    """
    Store feedback on a result.

    :param feedback_request: The feedback request.
    :param keyphrase: The keyphrase to get right config.

    :return: The status of the feedback storage.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    config = Config.get_config(keyphrase)
    file_id = feedback_request.file_id
    feedback = feedback_request.feedback

    feedback_dir = f"{config.api.savedir}/{file_id}"
    feedback_file = f"{feedback_dir}/feedback.json"
    os.makedirs(feedback_dir, exist_ok=True)
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as file:
                feedback_list = json.load(file)
        else:
            feedback_list = []
        feedback_list.append({"comment": feedback})
        with open(feedback_file, "w") as file:
            json.dump(feedback_list, file)
        return {
            "status": "success",
            "message": "Feedback stored successfully.",
        }
    except Exception as e:
        ulogger.info(e)
        return {
            "status": "error",
            "message": "An internal error has occurred.",
        }


@router.get("/api/get_feedback/{keyphrase}")
async def get_feedback(file_id: str, keyphrase: str, request: Request):
    """
    Get feedback on a result.

    :param file_id: The file id.
    :param keyphrase: The keyphrase to get right config.

    :return: The feedback on the result.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    config = Config.get_config(keyphrase)
    feedback_file = f"{config.api.savedir}/{file_id}/feedback.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as file:
            feedback_list = json.load(file)
        return {"status": "success", "feedback": feedback_list}
    return {"status": "error", "message": "Feedback not found."}


@router.get("/api/get_metrics/{keyphrase}")
async def get_metrics(file_id: str, keyphrase: str, request: Request):
    """
    Get metrics on a result.

    :param file_id: The file id.
    :param keyphrase: The keyphrase to get right config.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    config = Config.get_config(keyphrase)
    base_dir = config.api.savedir
    metrics_file = os.path.normpath(
        os.path.join(base_dir, file_id, "metrics.json")
    )
    if not metrics_file.startswith(os.path.abspath(base_dir)):
        return {"status": "error", "message": "Invalid file_id."}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as file:
            metrics_list = json.load(file)
            metrics_list["status"] = "success"
        return metrics_list
    return {"status": "error", "message": "metrics not found."}


@router.post("/api/upload/{keyphrase}")
async def handle_upload(
    request: Request, keyphrase: str, file: UploadFile = File(...)
):
    """
    Handle file uploads.

    :param keyphrase: The keyphrase to get right config.
    :param file: The file to upload.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    config = Config.get_config(keyphrase)

    os.makedirs(config.api.upload_dir, exist_ok=True)
    file_location = f"{config.api.upload_dir}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    if config.api.zero_trust:
        live_savedir = RESULTS_DIR
    else:
        live_savedir = config.api.savedir

    # check if the file is a zip, and extract it
    if file.filename.endswith(".zip"):
        # create a temp folder in the upload dir
        temp_dir = f"{config.api.upload_dir}/temp"

        shutil.unpack_archive(file_location, temp_dir)

        zip_name = file.filename.split(".")[0]

        # add the zips name to each extracted file
        for root, dirs, files in os.walk(temp_dir):
            for _file in files:
                os.rename(
                    os.path.join(root, _file),
                    os.path.join(root, f"{zip_name}_{_file}"),
                )

        # move the extracted files to the upload dir
        for root, dirs, files in os.walk(temp_dir):
            for _file in files:
                shutil.move(
                    os.path.join(root, _file),
                    os.path.join(config.api.upload_dir, _file),
                )

                dicom_id = _file.split(".")[0]
                await dicom_processing_queue.put(
                    (dicom_id, None, config, live_savedir)
                )

        # delete the temp folder
        shutil.rmtree(temp_dir)

        # delete the zip file
        os.remove(file_location)
        return {
            "info": f"file '{file.filename}' extracted at '{config.api.upload_dir}'"
        }

    dicom_id = file.filename.split(".")[0]
    await dicom_processing_queue.put((dicom_id, None, config, live_savedir))

    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@router.post(
    "/api/states/{keyphrase}",
    response_model=SystemFiles,
    responses={400: {"description": "Error"}},
)
@router.get(
    "/api/states/{keyphrase}",
    response_model=SystemFiles,
    responses={400: {"description": "Error"}},
)
async def get_states(keyphrase: str, request: Request):
    """
    Get the states from the database.

    :param keyphrase: The keyphrase to get right config.

    :return: The states from the database.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    config = Config.get_config(keyphrase)
    states = extract_files(config.api.db_path)
    output_states = SystemFiles(states=states, length=len(states))

    return output_states


@router.get("/api/keyphrases")
async def keyphrases(request: Request):
    """
    Get all valid keyphrases.

    :return: The valid keyphrases.
    """

    api_token = request.cookies.get("api_token")
    validate_api_token(api_token)

    keyphrases = [keyphrase for keyphrase, _ in Config.get_configs()]
    return JSONResponse(content={"keyphrases": keyphrases})
