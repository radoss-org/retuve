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
Class for creating visual overlays on images.
"""

from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from radstract.data.colors import LabelColours

from retuve.classes.seg import SegFrameObjects
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import Colors


class DrawTypes(Enum):
    """
    Enum for the different types of overlays that can be drawn.

    """

    LINES = "lines"
    SEGS = "segs"
    TEXT = "text"
    POINTS = "points"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"

    @classmethod
    def ALL(cls) -> List["DrawTypes"]:
        """
        Specified in order of drawing.
        """
        return [
            cls.SEGS,
            cls.LINES,
            cls.CIRCLE,
            cls.POINTS,
            cls.RECTANGLE,
            cls.TEXT,
        ]

    @classmethod
    def type_to_func(cls, draw, dtype):
        """
        Returns the function to use for a given overlay type.

        :param draw: ImageDraw object.
        :param dtype: Type of overlay to draw.

        :return: Function to use for a given overlay type.
        """
        if dtype == cls.LINES:
            return draw.line
        elif dtype == cls.SEGS:
            return draw.polygon
        elif dtype == cls.TEXT:
            return draw.text
        elif dtype == cls.POINTS:
            return draw.line
        elif dtype == cls.CIRCLE:
            return draw.ellipse
        elif dtype == cls.RECTANGLE:
            return draw.rectangle


class Overlay:
    """
    Class for creating visual overlays on images.
    """

    def __init__(self, shape: Tuple[int, int], config: Config):
        """
        :param shape: Shape of the image to overlay on.
        :param config: Config object.

        :attr shape: Shape of the image to overlay on.
        :attr overlays: Dictionary of overlays.
        :attr config: Config object.
        """
        self.shape = (shape[0], shape[1], 4)
        self.config = config
        self.operations = {}

        for drtype in DrawTypes.ALL():
            self.operations[drtype] = []

    def add_operation(self, optype, *args, **kwargs):
        # Store the drawing operation in the operations list
        self.operations[optype].append(((args, kwargs)))

    def apply_to_image(self, image):

        if not any(self.operations.values()):
            return image

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image, "RGBA")

        # Execute all stored operations
        # go in order of segs, lines, points, text
        for optype in DrawTypes.ALL():
            for args, kwargs in self.operations[optype]:
                func = DrawTypes.type_to_func(draw, optype)
                func(*args, **kwargs)

        final_image = np.array(image, dtype=np.uint8)

        # remove alpha channel
        final_image = final_image[:, :, :3]

        return final_image

    def get_nifti_frame(
        self, seg_frame_objs: SegFrameObjects, shape: list
    ) -> Image.Image:
        """
        Create an image with segmentation overlay ready for NIFTI storage.

        :param seg_frame_objs: Segmentation frame objects.
        :param shape: Shape of the image.

        :return: Image with segmentation overlay.
        """

        # Create empty frame of write shape
        seg_overlay = Image.new("RGB", shape[:2], (0, 0, 0))
        draw = ImageDraw.Draw(seg_overlay)

        for seg_obj in seg_frame_objs:
            if not seg_obj.empty:
                draw.polygon(
                    seg_obj.points,
                    fill=LabelColours.get_color_from_index(seg_obj.cls.value + 1),
                )

        return seg_overlay

    def draw_cross(self, point: Tuple[int, int], override_line_thickness: int = None):
        """
        Draws a cross of a given radius at a specified point on the overlay.

        :param point: Point to draw the cross at.
        """
        x, y = point

        radius = self.config.visuals.points_radius
        color = self.config.visuals.points_color.rgba()

        self.add_operation(
            DrawTypes.POINTS,
            (x - radius, y, x + radius, y),
            fill=color,
            width=override_line_thickness or self.config.visuals.line_thickness,
        )
        self.add_operation(
            DrawTypes.POINTS,
            (x, y - radius, x, y + radius),
            fill=color,
            width=override_line_thickness or self.config.visuals.line_thickness,
        )

    def draw_segmentation(self, points: List[Tuple[int, int]]):
        """
        Draws a polygon on the overlay.

        :param points: List of points to draw the polygon.
        """

        self.add_operation(
            DrawTypes.SEGS,
            points,
            fill=self.config.visuals.seg_color.rgba(self.config.visuals.seg_alpha),
        )

    def draw_box(self, box: Tuple[int, int, int, int], grafs: bool = False):
        """
        Draws a bounding box on the overlay.

        :param box: Bounding box to draw.
        :param grafs: Whether to draw with the graf frame colours or not.

        """
        x1, y1, x2, y2 = box
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))  # bottom right corner

        if grafs:
            color = self.config.visuals.graf_color.rgba()
            width = 10
        else:
            color = self.config.visuals.bounding_box_color.rgba()
            width = self.config.visuals.bounding_box_thickness

        self.add_operation(
            DrawTypes.RECTANGLE,
            [start_point, end_point],
            outline=color,
            width=width,
        )

    def draw_skeleton(self, skel: List[Tuple[int, int]]):
        """
        Draws a skeleton on the overlay.

        :param skel: List of points to draw the skeleton.
        """

        for point in skel:
            y, x = point
            self.add_operation(
                DrawTypes.POINTS,
                (x, y),
                fill=self.config.hip.midline_color.rgba(),
            )

    def draw_lines(
        self,
        line_points: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        color_override: Optional[Colors] = None,
    ):
        """
        Draws lines on the overlay.

        :param line_points: List of tuples of points to draw lines between.

        """

        if color_override:
            color = color_override
        else:
            color = self.config.visuals.line_color.rgba()

        for point1, point2 in line_points:
            self.add_operation(
                DrawTypes.LINES,
                (tuple(point1), tuple(point2)),
                fill=color,
                width=self.config.visuals.line_thickness,
            )

    def draw_text(
        self,
        label_text: str,
        x1: int,
        y1: int,
        header: Literal["h1", "h2"] = "h1",
        grafs: bool = False,
    ):
        """
        Draws text on the overlay.

        :param label_text: Text to draw.
        :param x1: x coordinate to draw text.
        :param y1: y coordinate to draw text.
        :param header: Header type to use.
        :param grafs: Whether to draw with the graf frame colours or not.

        """
        if header == "h1":
            font = self.config.visuals.font_h1
        elif header == "h2":
            font = self.config.visuals.font_h2

        temp_draw = ImageDraw.Draw(Image.new("RGBA", self.shape[:2]))

        bbox = temp_draw.textbbox((x1, y1), label_text, font=font)

        if grafs:
            color = self.config.visuals.graf_color.rgba()
        else:
            color = self.config.visuals.background_text_color.rgba()

        self.add_operation(
            DrawTypes.RECTANGLE,
            bbox,
            fill=color,
        )

        self.add_operation(
            DrawTypes.TEXT,
            (x1, y1),
            label_text,
            font=font,
            fill=self.config.visuals.text_color.rgba(),
        )

    def draw_circle(self, point: Tuple[int, int], radius: int):
        """
        Draws a circle on the overlay.

        :param point: Center of the circle.
        :param radius: Radius of the circle.
        """

        x, y = point

        self.add_operation(
            DrawTypes.CIRCLE,
            (x - radius, y - radius, x + radius, y + radius),
            outline=self.config.visuals.points_color.rgba(),
            width=self.config.visuals.line_thickness,
        )
