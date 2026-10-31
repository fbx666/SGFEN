# MUUFL Data

`SGFEN-MUUFL.py` reads its inputs from this directory (paths are relative to the
dataset folder), so the MUUFL dataset must be placed here before training.

## Required files

| File | MATLAB variable | Description |
| --- | --- | --- |
| `hsi_data.mat` | `hsi_data` | Hyperspectral image, `H × W × B` |
| `lidar_data.mat` | `lidar_data` | LiDAR elevation map, `H × W` (`H × W × 1` is also accepted, only the first channel is used) |
| `labels.mat` | `labels` | Ground truth, `H × W` (`0` = unlabeled); the script splits it into train/test |

The MUUFL Gulfport scene contains 11 land-cover classes.

## Download

Baidu Netdisk: <https://pan.baidu.com/s/1nXBUFR-KDa-q_npbkmMh1Q?pwd=wi62>
(password: `wi62`)

## How to place the files

After extracting the archive, move the three `.mat` files into this directory so the
layout becomes:

```
MUUFL/
├── SGFEN-MUUFL.py
├── utils.py
├── model/
└── data/
    ├── hsi_data.mat
    ├── lidar_data.mat
    └── labels.mat
```

Then run the script from the `MUUFL/` folder:

```
cd MUUFL
python SGFEN-MUUFL.py
```
