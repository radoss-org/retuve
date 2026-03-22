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
Code to control inserting/reading present state
of results into the database.
"""

import os
import sqlite3
from typing import List

from retuve.app.classes import File, FileEnum

# Get this files DIR
DIR = os.path.dirname(os.path.abspath(__file__))


def database_init(db_path: str):
    """
    Initialize the database.

    :param db_path: The path to the database.
    """
    # delete the database if it exists
    if not os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create the table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    state TEXT,
                    metrics_url TEXT,
                    video_url TEXT,
                    img_url TEXT,
                    figure_url TEXT,
                    attempts INTEGER DEFAULT 0
                )
                """)
            conn.commit()


def insert_files(files: List[File], db_path: str):
    """
    Insert files into the database.

    :param files: The list of files to insert.
    :param db_path: The path to the database.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for file in files:
            cursor.execute(
                """
                INSERT OR REPLACE INTO files (file_id, state, metrics_url, video_url, img_url, figure_url, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file.file_id,
                    file.state.value,
                    file.metrics_url,
                    file.video_url,
                    file.img_url,
                    file.figure_url,
                    file.attempts,
                ),
            )
            conn.commit()


# Extract function
def extract_files(db_path: str) -> List[File]:
    """
    Extract files from the database.

    :param db_path: The path to the database.
    :return: The list of files.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT file_id, state, metrics_url, video_url, img_url, figure_url, attempts FROM files"
            )
        except sqlite3.OperationalError:
            return []

        rows = cursor.fetchall()
        files = [
            File(
                file_id=row[0],
                state=FileEnum(row[1]),
                metrics_url=row[2],
                video_url=row[3],
                img_url=row[4],
                figure_url=row[5],
                attempts=row[6],
            )
            for row in rows
        ]
        conn.commit()

        files.sort(key=lambda x: (x.state.value, x.file_id))

        return files


def get_file_data(file_id: str, db_path: str) -> File:
    """
    Get the file data from the database.

    :param file_id: The file id.
    :param db_path: The path to the database.
    :return: The file data.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_id, state, metrics_url, video_url, img_url, figure_url, attempts FROM files WHERE file_id = ?",
            (file_id,),
        )
        row = cursor.fetchone()

        conn.commit()

        if not row:
            return File(
                file_id=file_id,
                state=FileEnum.PENDING,
                metrics_url="N/A",
                video_url="N/A",
                img_url="N/A",
                figure_url="N/A",
                attempts=0,
            )

        return File(
            file_id=row[0],
            state=FileEnum(row[1]),
            metrics_url=row[2],
            video_url=row[3],
            img_url=row[4],
            figure_url=row[5],
            attempts=row[6],
        )
