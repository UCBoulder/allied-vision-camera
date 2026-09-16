# allied-vision-camera

Snapshot-oriented Python wrapper around [`vmbpy`](https://github.com/alliedvision/VmbPy) for Allied Vision cameras (Alvium G1 family), plus PSF plotting helpers.

One canonical camera layer for the yb2 lab — pull it into any project as a git submodule instead of copying camera code around.

**Design:** camera configuration is slow and happens once at construction; snapshots are fast and never re-query hardware (all metadata is cached internally).

---

## Prerequisites

**The Vimba X SDK must be installed system-wide.** `vmbpy` is only a Python binding — it will not work without the vendor transport layer.

1. Download Vimba X from the [Allied Vision software page](https://www.alliedvision.com/en/products/software/).
2. Install it (this registers the GenTL transport layer).
3. Verify the camera is visible in Vimba X Viewer, then **close the Viewer** — it holds the camera in Full access mode and `AlviumG1` will refuse to open.

Python ≥ 3.11.

## Installation

As a standalone checkout:

```bash
git clone git@github.com:UCBoulder/allied-vision-camera.git
cd allied-vision-camera
pip install -e .
```

As a submodule in another project (the intended use):

```bash
git submodule add git@github.com:UCBoulder/allied-vision-camera.git external/allied_vision_camera
git submodule update --init --recursive
pip install -e external/allied_vision_camera
```

Adding the submodule path to `PYTHONPATH` also works, but `pip install -e` is cleaner and makes the import work from any working directory.

> Submodules pin a **commit**, not a branch. To pick up camera updates in a consuming project:
> `git submodule update --remote external/allied_vision_camera` and commit the moved pointer.

## Usage

```python
from allied_vision import AlviumG1

cam = AlviumG1({
    "data_root": "data",
    "exposure_us": 6000,
    "gain_db": 0.0,
})

img, out_dir = cam.snapshot(
    experiment_name="alignment",
    comment="after realigning L3",
    save_preview=True,
)

cam.close()
```

`img` is a `uint16` NumPy array (Mono12, values 0–4095). `out_dir` is the directory the snapshot was written to.

### Output layout

```
<data_root>/<experiment_name>/snapshot_<YYYY_MM_DD-HH_MM_SS>/
├── data_<stamp|filename>.npy       # raw uint16 frame
├── preview_<stamp|filename>.png    # if save_preview=True
├── preview_3d_<...>.png            # if plot_psf_3d=True
└── metadata_<stamp|filename>.json  # if save_metadata=True (default)
```

## Configuration

Passed as a dict to `AlviumG1(config)`. Only `data_root` is required; every other key falls back to the default in the `_PARAMETERS` registry.

| Key | Default | Meaning |
|---|---|---|
| `data_root` | *(required)* | Root directory for snapshot output. Created if missing. |
| `camera_id` | `None` | Specific camera ID. `None` connects to the first camera found. |
| `exposure_us` | `6000` | Exposure time in µs. Ignored if `exposure_auto` is `True`. |
| `gain_db` | `0.0` | Analog gain in dB. Ignored if `gain_auto` is `True`. |
| `exposure_auto` | `False` | Continuous auto-exposure. |
| `gain_auto` | `False` | Continuous auto-gain. |
| `pixel_format` | `"Mono12"` | `"Mono12"` or `"Mono8"`. |

## API

| | |
|---|---|
| `AlviumG1(config)` | Connect, open in Full access mode, apply settings, cache metadata. |
| `.snapshot(experiment_name, ...)` | Capture and save one frame. Returns `(img, out_dir)`. |
| `.capture_frame()` | Capture one frame in memory, no disk I/O. Returns `uint16` array. |
| `.plot(experiment_name=None, npy_path=None, cmap=None)` | Display a saved snapshot (defaults to the most recent one). |
| `.set_exposure(us)` / `.set_gain(db)` | Change settings at runtime; the metadata cache follows. |
| `.close()` | Reset to Mono8, close the camera, shut down the Vimba system. **Always call this** — an unclosed camera stays locked. |

`snapshot()` also accepts `path` (override the output directory), `filename` (fixed stem instead of a timestamp), `exposure_time_us` (one-shot exposure, restored afterwards), `cropping_params` (`{"offset_px": (x, y), "width_px": w, "height_px": h}`, AOI restored afterwards), `plot_psf_3d`, `save_preview`, `save_metadata`, and `comment`.

### Plotting helpers

Usable standalone on any 2-D array — no camera required:

```python
from allied_vision import save_psf_2d, save_psf_3d

save_psf_2d(img, "preview.png", title="PSF")   # side-by-side fixed 0–4095 / auto-scaled
save_psf_3d(img, "preview_3d.png", title="PSF")  # 3-D surface
```

## Known constraints

- **12-bit / Mono12 is assumed throughout.** The `UINT12_MAX = 4095` ceiling is baked into the plot scaling, and `capture_frame()` takes channel 0 of the converted frame.
- Frame timeout is fixed at 2000 ms.
- Status messages go to `stdout` via `print()`, not the `logging` module.
- Only one camera is opened per `AlviumG1` instance; with no `camera_id` it takes whichever camera enumerates first.

## Provenance

Extracted from `hardware/camera/` in [UCBoulder/DMD](https://github.com/UCBoulder/DMD) as a standalone, reusable package. That repo keeps its own copy for now; this one is the canonical version going forward.
