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
The parent class for for Retuve configs
"""

import copy
from typing import List, Tuple

from PIL import ImageFont
from radstract.data.dicom import DicomTypes

from retuve.keyphrases.enums import OperationType
from retuve.keyphrases.subconfig import (
    APIConfig,
    BatchConfig,
    HipConfig,
    TrakConfig,
    VisualsConfig,
)
from retuve.logs import ulogger
from retuve.utils import RETUVE_DIR, register_config_dirs


class GlobalConfig:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password


class Config:
    """Configuration class."""

    configs = {}
    live_config = None

    def __init__(
        self,
        dicom_type: DicomTypes,
        crop_coordinates: Tuple[float],
        template: bool,
        min_seg_confidence: float,
        device: "torch.Device",  # type: ignore
        operation_type: OperationType,
        dev: bool,
        replace_old: bool,
        seg_export: bool,
        test_data_passthrough: bool,
        subconfig_hip: HipConfig,
        subconfig_trak: TrakConfig,
        subconfig_visuals: VisualsConfig,
        subconfig_api: APIConfig,
        subconfig_batch: BatchConfig,
        name=None,
    ):
        """
        Initialize Config.

        :param dicom_type (DicomTypes): The type of dicom to use.
        :param crop_coordinates (Tuple[float]): The crop coordinates.
        :param template (bool): Whether this is a template config.
        :param min_seg_confidence (float): The minimum segmentation confidence.
        :param device (Device): The device to use.
        :param operation_type (OperationType): The operation type.
        :param dev (bool): Whether to use dev mode.
        :param replace_old (bool): Whether to replace old files.
        :param seg_export (bool): Whether to export segmentation.
        :param test_data_passthrough (bool): Whether to pass through test data.
        :param subconfig_hip (HipConfig): The hip subconfig.
        :param subconfig_trak (TrakConfig): The trak subconfig.
        :param subconfig_visuals (VisualsConfig): The visuals subconfig.
        :param subconfig_api (APIConfig): The api subconfig.
        :param subconfig_batch (BatchConfig): The batch subconfig.
        :param name (str): The name (keyphrase) of the config.
        """
        self.name = name

        if not (template or name):
            raise ValueError("Template configs must have a name specified.")

        if not template and (name):
            self.register(name)

        self.dicom_type = dicom_type
        self.crop_coordinates = crop_coordinates
        self.min_seg_confidence = min_seg_confidence

        self.device = device
        self.replace_old = replace_old
        self.operation_type = operation_type
        self.dev = dev

        self.seg_export = seg_export

        self.hip: HipConfig = subconfig_hip
        self.trak: TrakConfig = subconfig_trak
        self.visuals: VisualsConfig = subconfig_visuals
        self.api: APIConfig = subconfig_api
        self.batch: BatchConfig = subconfig_batch

        self.test_data_passthrough = test_data_passthrough

        if operation_type not in [
            OperationType.SEG,
            OperationType.LANDMARK,
        ]:
            raise ValueError(f"Invalid operation type: {operation_type}")

    def register(
        self,
        name: str,
        store: bool = True,
        silent: bool = False,
        live: bool = False,
    ):
        """
        Register the config.

        :param name (str): The name of the config.
        :param store (bool): Whether to store the config
                             for later retrieval.

        """
        self.name = name

        # check if config already exists
        if self.configs.get(name):
            raise ValueError(f"Config {name} already exists.")

        if not self.configs.get(name) and store:
            self.configs[name] = {}

        if store:
            self.configs[name] = self

        if live:
            Config.live_config = self

        # defaults need registering
        if self.visuals.default_font_size:
            self.visuals.font_h1 = ImageFont.truetype(
                f"{RETUVE_DIR}/files/RobotoMono-Regular.ttf",
                self.visuals.default_font_size,
            )

            self.visuals.font_h2 = ImageFont.truetype(
                f"{RETUVE_DIR}/files/RobotoMono-Regular.ttf",
                self.visuals.default_font_size,
            )

        self.batch.register()

        register_config_dirs(self)

        if not self.api.api_token:
            ValueError("API token must be set.")

        if not silent:
            ulogger.info(f"Registered config for {name}")

    def unregister(self, silent: bool = False):
        """
        Unregister the config.
        """
        if self.name in self.configs:
            del self.configs[self.name]

        if not silent:
            ulogger.info(f"Unregistered config for {self.name}")

    def get_copy(self) -> "Config":
        """
        Get a copy of the config.

        :return: A copy of the config.
        """
        return copy.deepcopy(self)

    def inject_global_config(self, username, password):
        """
        Inject the global config into the config.

        :param username (str): The username.
        :param password (str): The password.
        """
        Config.global_config = GlobalConfig(username, password)

    @classmethod
    def get_configs(self) -> List[Tuple[str, "Config"]]:
        """
        Get all registered/stored configs.

        :return: A list of tuples of name and config.
        """
        return [(name, config) for name, config, in self.configs.items()]

    @classmethod
    def get_config(cls, name: str) -> "Config":
        """
        Get a config by name.

        :param name (str): The name of the config.

        :return: The config.
        """
        # check if name is of type Config
        if isinstance(name, Config):
            return name

        if not cls.configs.get(name):
            raise ValueError(f"Config {name} does not exist.")

        return cls.configs[name]

    @classmethod
    def keyphrase_exists(cls, name: str) -> bool:
        """
        Check if a keyphrase exists in the configs.

        :param name (str): The name of the keyphrase.

        :return: Whether the keyphrase exists.
        """
        lower_keys = [key.lower() for key in cls.configs.keys()]
        return name.lower() in lower_keys
