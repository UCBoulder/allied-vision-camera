"""
Allied Vision camera control for the yb2 lab.

Snapshot-oriented wrapper around vmbpy, plus PSF plotting helpers.

Usage
-----
from allied_vision import AlviumG1

cam = AlviumG1({"data_root": "data", "exposure_us": 6000, "gain_db": 0.0})
img, path = cam.snapshot(experiment_name="alignment", save_preview=True)
cam.close()
"""

from .alviumG1_v1 import AlviumG1
from .camera_plot_helpers import PSF_CMAP, UINT12_MAX, save_psf_2d, save_psf_3d

__all__ = ["AlviumG1", "save_psf_2d", "save_psf_3d", "PSF_CMAP", "UINT12_MAX"]

__version__ = "0.1.0"
