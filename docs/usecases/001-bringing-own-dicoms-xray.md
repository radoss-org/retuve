# Bringing Your Own X-Ray DDH DICOMs

This guide will show you how to analyze your own X-Ray DICOMs using Retuve.

## Install Retuve's AI Plugins

As of 2025-06-01, Retuve does not include default AI plugins. You will need to install an external plugin. The following plugins are available:

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.

- [retuve-yolo-plugin](https://github.com/radoss-org/retuve-yolo-plugin)

For this guide, we will use the `retuve-yolo-plugin`.

Install it with:

```bash
pip install retuve-yolo-plugin
```

We provide an up-to-date list of examples for using Retuve with X-Rays [here](https://github.com/radoss-org/retuve/tree/main/examples/x-ray). The most useful are:
- [Running Retuve with X-ray jpegs/pngs](https://github.com/radoss-org/retuve/tree/main/examples/x-ray/ai_with_jpeg.py)
- [Running Retuve with X-ray dicoms](https://github.com/radoss-org/retuve/tree/main/examples/x-ray/ai_with_dicom.py)
- [Running Retuve against a folder of X-rays](https://github.com/radoss-org/retuve/tree/main/examples/x-ray/ai_with_multiple_jpgs.py)

We recommend that you change the following in these scripts:
- Set the device with `my_config.device` or `default_xray.device` if you are using `cuda`

## Further Reading

- [Retuve Documentation](https://retuve.nidusai.ca/retuve.html)
- [YOLO Plugin for Retuve](https://github.com/radoss-org/retuve-yolo-plugin)

---

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.