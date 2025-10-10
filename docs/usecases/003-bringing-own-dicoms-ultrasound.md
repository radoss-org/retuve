# Bringing Your Own Ultrasound DICOMs

This guide will show you how to analyze your own Ultrasound DICOMs using Retuve. Retuve supports three types of ultrasound analysis:
- 2D Ultrasound (single frame)
- 3D Ultrasound (volume)
- 2D Sweep Ultrasound (multiple frames)

## Install Retuve's AI Plugins

As of 2025-06-01, Retuve does not include default AI plugins. You will need to install an external plugin. The following plugins are available:

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.

- [retuve-tinyunet-plugin](https://github.com/radoss-org/retuve-tinyunet-plugin)
- [retuve-nnunet-plugin](https://github.com/radoss-org/retuve-nnunet-plugin)
- [retuve-yolo-plugin](https://github.com/radoss-org/retuve-yolo-plugin)

For this guide, we will use the `retuve-yolo-plugin`.

Install it with:

```bash
pip install retuve-yolo-plugin
```

We provide an up-to-date list of examples for using Retuve with X-Rays [here](https://github.com/radoss-org/retuve/tree/main/examples/ultrasound). The most useful are:
- [Running Retuve with 2D ultrasound jpegs/pngs](https://github.com/radoss-org/retuve/tree/main/examples/ultrasound/ai_with_jpeg_2d.py)
- [Running Retuve with 3D/Sweep ultrasound dicoms](https://github.com/radoss-org/retuve/tree/main/examples/ultrasound/ai_with_dicom.py)
- [Running Retuve against a folder of ultrasounds](https://github.com/radoss-org/retuve/tree/main/examples/ultrasound/ai_with_multiple_scans.py)

We recommend that you change the following in these scripts:
- Set the device with `my_config.device` or `default_US.device` if you are using `cuda`


## Further Reading

- [Retuve Documentation](https://retuve.nidusai.ca/retuve.html)
- [YOLO Plugin for Retuve](https://github.com/radoss-org/retuve-yolo-plugin)

---

**Note:**
All plugins are under their own licenses. Please check the license of each plugin before using it.

---

If you need to process individual files or want more control, see the [minimal script guide](https://github.com/radoss-org/retuve/tree/main/examples/high_level_functions). example in the other part of this codebase.