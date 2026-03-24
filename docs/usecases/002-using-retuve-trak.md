# Using Retuve Trak with X-Ray DICOM Files

This guide explains how to run a batch analysis of your own X-Ray DICOM files using Retuve, and view them in the Retuve Trak Web App.

**Note**
Retuve Trak is capable of more: for example, connecting directly to a PACS server. However, currently these features are in beta. For more information, please contact the [Maintainers](https://github.com/radoss-org/retuve#maintainers).

**Note:**
All plugins are under their own licenses. Please check the license of the plugin before using it.

## Prerequisites

- **YOLO plugin for Retuve**:
  Install with:
  ```bash
  pip install retuve-yolo-plugin
  ```
- Your X-Ray DICOM files placed in a directory (e.g., `./your/path/`).

## What Does This Script Do?

- Loads the YOLO model for X-Ray analysis.
- Configures Retuve to process all DICOM files in a given directory.
- Runs a batch analysis using the YOLO model.
- Initializes the Retuve web app and state machines.
- Serves the results via the Retuve Trak server.

## Usage

### 1. Prepare Your DICOM Files

Place all your DICOM files in a directory, for example:
`./your/path/`

### 2. Save the Script

Save the following script as `run_batch_xray.py`:

```python
import os

import uvicorn
from retuve_yolo_plugin.xray import (
    get_yolo_model_xray,
    yolo_predict_dcm_xray,
    yolo_predict_xray,
)

from retuve.app import app
from retuve.app.helpers import app_init
from retuve.batch import run_batch
from retuve.defaults.hip_configs import default_xray
from retuve.keyphrases.enums import HipMode
from retuve.testdata import Cases, download_case
from retuve.trak.main import run_all_state_machines

# or DATASET_DIR = "path/to/files"
jpg_file, *_ = download_case(Cases.XRAY_JPG_DATASET)
DATASET_DIR = os.path.join(os.path.dirname(jpg_file))

my_config = default_xray.get_copy()
my_config.batch.hip_mode = HipMode.XRAY
my_config.batch.mode_func = yolo_predict_xray
my_config.batch.mode_func_args = {"model": get_yolo_model_xray(my_config)}
my_config.batch.input_types = [".jpg"]
my_config.batch.datasets = [DATASET_DIR] # or ["path/to/files"]
my_config.api.savedir = "./.test-output"
my_config.device = "cpu"

def setup():
    my_config.register("My Config Name")
    my_config.inject_global_config("username", "password")

if __name__ == "__main__":
    setup()
    run_batch(my_config)
    app_init(app)
    run_all_state_machines()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Configure Your Credentials

Edit the `setup()` function in the script to use your own username and password if needed:

```python
my_config.inject_global_config("username", "password")
```

### 4. Run the Script

From your terminal, run:

```bash
python run_batch_xray.py
```

- The script will process all DICOM files in `./your/path/`.
- The Retuve Trak server will start on `http://0.0.0.0:8000`.


### 5. Access the Retuve Trak Web App

Open your web browser and navigate to `http://localhost:8000` (or the IP address of the server if running remotely). You will see the Retuve Trak interface where you can view the results of your batch analysis.

![https://files.mcaq.me/r26x7.png](https://files.mcaq.me/r26x7.png)

Click the "My Config Name - View History" button to see the results of your batch processing.

![https://files.mcaq.me/006h2.png](https://files.mcaq.me/006h2.png)

**Note**
Please ignore other buttons in the interface, as they are beta features and may not work as expected.

## Customization

- **Change the dataset directory:**
  Modify the line:
  ```python
  my_config.batch.datasets = ["./your/path/"]
  ```
  to point to your own directory of DICOM files.

- **Model or plugin:**
  You can swap out the YOLO plugin for another compatible Retuve plugin if desired:

  - https://github.com/radoss-org/retuve-tinyunet-plugin
  - https://github.com/radoss-org/retuve-nnunet-plugin

## Troubleshooting

## Further Reading

- [Retuve Documentation](https://retuve.nidusai.ca/retuve.html)
- [YOLO Plugin for Retuve](https://github.com/radoss-org/retuve-yolo-plugin)

---

**Note:**
All plugins are under their own licenses. Please check the license of the plugin before using it.

---

If you need to process individual files or want more control, see the [minimal script guide](https://github.com/radoss-org/retuve/blob/main/examples/high_level_functions/xray.py) or the example in the other part of this codebase.