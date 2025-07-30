# Bringing Your Own X-Ray DDH DICOMs

This guide will show you how to analyze your own X-Ray DICOMs using Retuve.

## Install Retuve's AI Plugins

As of 2025-06-01, Retuve does not include default AI plugins. You will need to install an external plugin. The following plugins are available:

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.

- [retuve-yolo-plugin](https://github.com/radoss-org/retuve-yolo-plugin)

For this guide, we will use the `retuve-yolo-plugin`, which has the version 2 x-ray system.

**We highly recommend reading how version 2 was trained and validated here: https://github.com/radoss-org/retuve-yolo-plugin?tab=readme-ov-file#update---x-ray-version-2**

Install it with:

```bash
pip install git+https://github.com/radoss-org/retuve-yolo-plugin
```

## Creating a Minimal Script

You only need to make a few changes to the [example script](https://github.com/radoss-org/retuve/blob/main/examples/high_level_functions/xray.py) to run the AI on your own data.

### 1. Import the X-Ray Plugin and pydicom

```python
from retuve_yolo_plugin.xray_v2 import yolo_predict_dcm_xray
import pydicom
```

### 2. Load Your Own DICOM File

Replace the example code that loads a sample image and labels:

```python
img_file, labels_json = download_case(Cases.XRAY_JPG)
img_raw = Image.open(img_file)
labels = json.load(open(labels_json))
```

with:

```python
dcm_file = "path/to/your/dicom/file.dcm"
dcm = pydicom.dcmread(dcm_file)
```

You can remove all references to `labels_json`—it is not needed for the AI plugin.

### 3. Update the `analyse_hip_xray_2D` Function Call

Make sure you use the DICOM data and the plugin function:

```python
hip_data, img, dev_metrics = analyse_hip_xray_2D(
    dcm,
    keyphrase=default_xray,
    modes_func=yolo_predict_dcm_xray,
    modes_func_kwargs_dict={},
)
```

## Example Scripts

Here is a complete example script for analyzing a single DICOM file:

```python
import pydicom
from retuve_yolo_plugin.xray_v2 import yolo_predict_dcm_xray
from retuve.defaults.hip_configs import default_xray
from retuve.funcs import analyse_hip_xray_2D

# Example usage
dcm_file = "path/to/your/dicom/file.dcm"
dcm = pydicom.dcmread(dcm_file)

hip_data, img, dev_metrics = analyse_hip_xray_2D(
    dcm,
    keyphrase=default_xray,
    modes_func=yolo_predict_dcm_xray,
    modes_func_kwargs_dict={},
)

img.save("xray.jpg")

metrics = hip_data.json_dump(default_xray, dev_metrics)
print(metrics)
```

To process multiple DICOM files, use a loop:

```python
import os
import pydicom
from retuve_yolo_plugin.xray_v2 import yolo_predict_dcm_xray
from retuve.defaults.hip_configs import default_xray
from retuve.funcs import analyse_hip_xray_2D

dicom_files = [
    "path/to/your/dicom/file1.dcm",
    "path/to/your/dicom/file2.dcm",
    # Add more files as needed
]

for dcm_file in dicom_files:
    dcm = pydicom.dcmread(dcm_file)

    hip_data, img, dev_metrics = analyse_hip_xray_2D(
        dcm,
        keyphrase=default_xray,
        modes_func=yolo_predict_dcm_xray,
        modes_func_kwargs_dict={},
    )

    img.save(os.path.splitext(dcm_file)[0] + "_xray.jpg")

    metrics = hip_data.json_dump(default_xray, dev_metrics)
    print(metrics)
```

## Troubleshooting

- **Plugin not found:**
  Make sure you have installed the plugin as described above.
- **DICOM files not detected:**
  Ensure your files are in the correct location and are valid DICOM files.

## Further Reading

- [Retuve Documentation](https://retuve.nidusai.ca/retuve.html)
- [YOLO Plugin for Retuve](https://github.com/radoss-org/retuve-yolo-plugin)

---

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.

---

If you need to process individual files or want more control, see the [minimal script guide](https://github.com/radoss-org/retuve/blob/main/examples/high_level_functions/xray.py) or the example in the other part of this codebase.