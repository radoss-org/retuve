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

import asyncio
import hashlib
import json
import os
import secrets
from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
)
from httpx import AsyncClient

from retuve.funcs import retuve_run


def generate_token():
    return secrets.token_urlsafe(32)


def consistent_hash(value, mod):
    # Convert the input to a string and encode it
    encoded_value = str(value).encode()
    # Use a reproducible hash function (e.g., SHA-256)
    hash_value = int(hashlib.sha256(encoded_value).hexdigest(), 16)
    return hash_value % mod


TOKEN_STORE = {}
API_TOKEN_STORE = {}


async def save_dicom_and_get_results(live_batchdir, instance_id, dicom_content, config):
    # Save the DICOM file in the appropriate directory
    dicom_path = f"{live_batchdir}/{instance_id}.dcm"
    if dicom_content is not None:
        # if path already exists, return
        if os.path.exists(dicom_path):
            return
        with open(dicom_path, "wb") as f:
            f.write(dicom_content)

    # Process the DICOM
    result = await asyncio.to_thread(
        retuve_run,
        hip_mode=config.batch.hip_mode,
        config=config,
        modes_func=config.batch.mode_func,
        modes_func_kwargs_dict={},
        file=dicom_path,
    )

    return result


async def save_results(instance_id, savedir, result=None, just_paths=False):
    """
    Saves DICOM content, video, image, and metrics results for a given instance ID.

    :param instance_id: The unique ID of the DICOM instance.
    :param result: The result object returned by `retuve_run`.
    :param savedir: The base directory for saving the results.
    :param just_paths: Whether to return just the paths of the saved results.
    """
    result_dir = f"{savedir}/{instance_id}"
    os.makedirs(result_dir, exist_ok=True)

    video_path = f"{result_dir}/video.mp4"
    img_path = f"{result_dir}/img.jpg"
    metrics_path = f"{result_dir}/metrics.json"

    if just_paths:
        return video_path, img_path, metrics_path

    # Save video result if available
    if result.video_clip:
        await asyncio.to_thread(result.video_clip.write_videofile, video_path)

    # Save image result if available
    if result.image:
        await asyncio.to_thread(result.image.save, img_path)

    # Save metrics if available
    if result.metrics:
        with open(metrics_path, "w") as f:
            json.dump(result.metrics, f)

    return video_path, img_path, metrics_path


async def get_sorted_dicom_images(
    orthanc_url, username=None, password=None, latest_time=None
):
    """
    Fetch and sort DICOM images from an Orthanc server based on acquisition time.

    :param orthanc_url: The Orthanc server URL.
    :param username: Username for authentication (optional).
    :param password: Password for authentication (optional).
    :param latest_time: Filter for acquisition times after this datetime.
    :return: A tuple of (sorted list of (DICOM content, instance ID), latest acquisition datetime).
    """

    auth = (username, password) if username and password else None
    images_with_dates = []
    latest_acq_time = latest_time or datetime.min

    async with AsyncClient() as client:
        patients_response = await client.get(f"{orthanc_url}/patients", auth=auth)
        for patient_id in patients_response.json():
            studies_response = await client.get(
                f"{orthanc_url}/patients/{patient_id}/studies", auth=auth
            )
            for study in studies_response.json():
                study_id = study["ID"]
                series_response = await client.get(
                    f"{orthanc_url}/studies/{study_id}/series", auth=auth
                )
                for series in series_response.json():
                    series_id = series["ID"]
                    instances_response = await client.get(
                        f"{orthanc_url}/series/{series_id}/instances",
                        auth=auth,
                    )
                    for instance in instances_response.json():
                        instance_id = instance["ID"]
                        metadata_response = await client.get(
                            f"{orthanc_url}/instances/{instance_id}/simplified-tags",
                            auth=auth,
                        )
                        metadata = metadata_response.json()

                        # Remove files that are not multiframe US's
                        if not (
                            metadata.get("SOPClassUID") == "1.2.840.10008.5.1.4.1.1.3.1"
                            and int(metadata.get("NumberOfFrames", 0)) > 1
                        ):
                            continue

                        acq_date, acq_time = metadata.get(
                            "AcquisitionDate"
                        ), metadata.get("AcquisitionTime")
                        if acq_date and acq_time:
                            acq_datetime = datetime.strptime(
                                f"{acq_date}{acq_time.split('.')[0]}",
                                "%Y%m%d%H%M%S",
                            )

                            images_with_dates.append(
                                (acq_datetime, instance_id, metadata)
                            )

        final_images_with_dates = []

        if len(images_with_dates) == 0:
            return [], latest_acq_time

        for acq_datetime, instance_id, metadata in images_with_dates:
            if acq_datetime > latest_acq_time:
                file_response = await client.get(
                    f"{orthanc_url}/instances/{instance_id}/file",
                    auth=auth,
                )

                # hash the main instance_id to be smaller
                small_id = consistent_hash(instance_id, 10**8)
                patient_id_real = metadata.get("PatientID")
                instance_id = f"{acq_datetime.strftime('%Y-%m-%d-%H:%M:%S')} Patient-{patient_id_real} ID-{small_id}"

                latest_acq_time = max(latest_acq_time, acq_datetime)

                final_images_with_dates.append(
                    (acq_datetime, file_response.content, instance_id)
                )

        images_with_dates.sort(key=lambda x: x[0])
        latest_image = images_with_dates[-1]

        # inject file_response.content into latest_image
        file_response = await client.get(
            f"{orthanc_url}/instances/{latest_image[1]}/file",
            auth=auth,
        )

        instance_id = latest_image[1]
        acq_datetime = latest_image[0]
        metadata = latest_image[2]
        small_id = consistent_hash(instance_id, 10**8)
        patient_id_real = metadata.get("PatientID")
        instance_id = f"{acq_datetime.strftime('%Y-%m-%d-%H:%M:%S')} Patient-{patient_id_real} ID-{small_id}"

        latest_image = (
            acq_datetime,
            file_response.content,
            instance_id,
        )

        if not final_images_with_dates:
            final_images_with_dates = [latest_image]

    return [
        (image, instance_id) for _, image, instance_id in final_images_with_dates
    ], latest_acq_time


class UnauthorizedException(Exception):
    """Raised when the user is unauthorized and should be redirected."""

    pass


def validate_auth_token(auth_token: str):
    token_data = TOKEN_STORE.get(auth_token)
    if not token_data or token_data["expires"] < datetime.utcnow():
        raise UnauthorizedException
    return token_data["username"]


def validate_api_token(api_token: str):
    token_data = API_TOKEN_STORE.get(api_token)
    if not token_data or token_data["expires"] < datetime.utcnow():
        raise HTTPException(status_code=403, detail="Invalid API token.")
    return token_data["username"]
