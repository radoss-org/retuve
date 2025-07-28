import base64
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timedelta
from hashlib import sha256
from tempfile import mkdtemp

from cycler import V
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, JSONResponse

from retuve.app.helpers import API_RESULTS_URL_ACCESS, web_templates
from retuve.app.utils import (
    API_TOKEN_STORE,
    TOKEN_STORE,
    generate_token,
    get_sorted_dicom_images,
    save_dicom_and_get_results,
    save_results,
    validate_api_token,
    validate_auth_token,
)
from retuve.keyphrases.config import Config
from retuve.trak.data import extract_files


def basic_auth_dependency(
    authorization: str = Header(None), response: Response = None
):
    if not authorization or not authorization.startswith("Basic "):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    try:
        valid_username = Config.global_config.username
        valid_password = Config.global_config.password

        auth_token = authorization.split(" ")[1]
        decoded_credentials = base64.b64decode(auth_token).decode("utf-8")
        username, password = decoded_credentials.split(":")
        if username != valid_username or password != valid_password:
            raise ValueError("Invalid credentials")

        # Generate authentication token
        auth_token = generate_token()
        api_token = generate_token()

        # Store tokens with expiration timestamps
        expiration = datetime.utcnow() + timedelta(hours=24)
        TOKEN_STORE[auth_token] = {
            "username": username,
            "expires": expiration,
        }
        API_TOKEN_STORE[api_token] = {
            "username": username,
            "expires": expiration,
        }

        return {"auth_token": auth_token, "api_token": api_token}

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def web(
    request: Request,
    tokens: dict = Depends(basic_auth_dependency),
):
    """
    Open the main page of the Retuve Web Interface.
    """
    keyphrases = [keyphrase for keyphrase in Config.configs.keys()]

    response = web_templates.TemplateResponse(
        f"index.html",
        {
            "request": request,
            "keyphrases": keyphrases,
        },
    )

    response.set_cookie(
        "auth_token",
        tokens["auth_token"],
        expires=3600,
        httponly=True,
        secure=True,
        samesite="Strict",
    )
    response.set_cookie(
        "api_token",
        tokens["api_token"],
        expires=3600,
        httponly=True,
        secure=True,
        samesite="Strict",
    )

    return response


@router.get("/ui/live/", response_class=HTMLResponse)
async def live_ui(request: Request):
    """
    Open the live page of the Retuve Web Interface.
    """
    auth_token = request.cookies.get("auth_token")
    validate_auth_token(auth_token)

    return web_templates.TemplateResponse(
        f"live.html",
        {
            "request": request,
            "keyphrase": Config.live_config.name,
            "url": Config.live_config.api.url,
        },
    )


@router.get("/ui/{keyphrase}", response_class=HTMLResponse)
async def web_keyphrase(request: Request, keyphrase: str):
    """
    Open the page for a specific keyphrase.
    """

    try:
        config = Config.get_config(keyphrase)
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Keyphrase {keyphrase} not found"
        )
    files_data = extract_files(config.api.db_path)

    auth_token = request.cookies.get("auth_token")
    validate_auth_token(auth_token)

    return web_templates.TemplateResponse(
        f"main.html",
        {
            "request": request,
            "files": files_data,
            "hip_mode": config.batch.hip_mode,
            "keyphrase": keyphrase,
            "url": config.api.url,
        },
    )


@router.get("/ui/upload/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """
    Open the upload form for the Retuve Web Interface.
    """

    auth_token = request.cookies.get("auth_token")
    validate_auth_token(auth_token)

    keyphrases = [keyphrase for keyphrase in Config.configs.keys()]

    return web_templates.TemplateResponse(
        f"upload.html",
        {
            "request": request,
            "keyphrases": keyphrases,
        },
    )


@router.get("/ui/download/{keyphrase}")
async def download_files(
    request: Request, keyphrase: str, pattern: str = None
):
    """
    Download files from the Retuve Web Interface.
    """

    auth_token = request.cookies.get("auth_token")
    validate_auth_token(auth_token)

    if not pattern:
        raise HTTPException(status_code=400, detail="Pattern is required")

    config = Config.get_config(keyphrase)
    savedir = config.api.savedir

    # Create a temporary directory to hold the folders
    temp_dir = mkdtemp()

    # Find all folders in savedir that match the pattern
    # Copy each matching folder to the temporary directory
    for folder in os.listdir(savedir):
        safe_pattern = re.escape(pattern)
        if re.search(safe_pattern, folder) and os.path.isdir(
            os.path.join(savedir, folder)
        ):
            source_folder = os.path.join(savedir, folder)
            destination_folder = os.path.join(temp_dir, folder)
            shutil.copytree(source_folder, destination_folder)

    # Create a zip file of the temporary directory
    zip_file = os.path.join(savedir, f"{pattern}.zip")
    shutil.make_archive(zip_file[:-4], "zip", temp_dir)

    zip_file_url = (
        f"{config.api.url}/{API_RESULTS_URL_ACCESS}/{keyphrase}/{pattern}.zip"
    )

    # Optionally, remove the temporary directory after creating the zip file
    shutil.rmtree(temp_dir)

    return web_templates.TemplateResponse(
        "download.html",
        {
            "request": request,
            "zip_file_url": zip_file_url,
            "keyphrase": keyphrase,
        },
    )
