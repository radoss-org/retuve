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
Default Configs for Each Subconfig in the Keyphrases Module.
"""

from radstract.data.dicom import DicomTypes
from retuve.keyphrases.config import Config
from retuve.keyphrases.enums import (
    ACASplit,
    Colors,
    CRFem,
    Curvature,
    GrafSelectionMethod,
    MetricUS,
    MidLineMove,
    OperationType,
)
from retuve.keyphrases.subconfig import (
    APIConfig,
    BatchConfig,
    HipConfig,
    TrakConfig,
    VisualsConfig,
)

RETUVE_DIR = "./retuve-data"

visuals = VisualsConfig(
    font_h1=None,
    font_h2=None,
    default_font_size=12,
    seg_color=Colors(Colors.PURPLE),
    seg_alpha=0.4,
    bounding_box_color=Colors(Colors.BLUE),
    bounding_box_thickness=2,
    text_color=Colors(Colors.WHITE),
    background_text_color=Colors(Colors.BLUE),
    points_color=Colors(Colors.LIGHT_BLUE),
    points_radius=6,
    hip_color=Colors(Colors.BLUE),
    line_color=Colors(Colors.WHITE),
    graf_color=Colors(Colors.GOLD),
    line_thickness=3,
    display_full_metric_names=False,
    display_boxes=False,
    display_segs=False,
    min_vid_length=6,
    min_vid_fps=30,
)

hip = HipConfig(
    midline_color=Colors(Colors.WHITE),
    aca_split=ACASplit.GRAFS,
    cr_fem_pos=CRFem.CENTER,
    fem_extention=0.1,
    midline_move_method=MidLineMove.BASIC,
    curvature_method=Curvature.RADIST,
    measurements=[
        MetricUS.ALPHA,
        MetricUS.COVERAGE,
        # MetricUS.CENTERING_RATIO,
        # MetricUS.ACA,
    ],
    z_gap=2.5,
    display_frame_no=True,
    display_side=False,
    draw_midline=False,
    display_fem_guess=False,
    draw_side_metainfo=False,
    allow_flipping=False,
    display_bad_frame_reasons=False,
    graf_selection_method=GrafSelectionMethod.MANUAL_FEATURES,
    graf_selection_func=None,
    graf_selection_func_args={},
    display_graf_conf=False,
    graf_algo_threshold=None,
    graf_frame_selection=None,
    allow_irregular_illiums=False,
    allow_horizontal_flipping=False,
    full_metric_functions=[],
    seg_preprocess_functions=[],
    per_frame_metric_functions=[],
    post_draw_functions=[],
    use_polyfit_replace_apex=False,
    count_0_coverage_as_error=False,
)

trak = TrakConfig(
    datasets=[
        f"{RETUVE_DIR}/default/uploaded",
    ],
)

api = APIConfig(
    savedir=f"{RETUVE_DIR}/default/savedir",
    url="http://localhost:8000",
    db_path=f"{RETUVE_DIR}/default/trak.db",
    upload_dir=f"{RETUVE_DIR}/default/uploaded",
    hippa_logging_file=f"{RETUVE_DIR}/hippa.log",
    api_token=None,
    origins=["http://localhost:8000"],
    zero_trust=True,
    zero_trust_interval=300,
    orthanc_url="http://localhost:8042",
    orthanc_username="orthanc",
    orthanc_password="orthanc",
)

batch = BatchConfig(
    mode_func=None,
    hip_mode=None,
    mode_func_args={},
    processes=1,
    input_types=[".dcm", ".jpg"],
    datasets=[api.upload_dir],
)

base_config = Config(
    dicom_type=DicomTypes.SERIES,
    template=True,
    crop_coordinates=None,
    min_seg_confidence=0.6,
    device="cpu",
    operation_type=OperationType.SEG,
    dev=False,
    replace_old=False,
    seg_export=False,
    subconfig_hip=hip,
    subconfig_trak=trak,
    subconfig_visuals=visuals,
    subconfig_api=api,
    subconfig_batch=batch,
    test_data_passthrough=False,
)
