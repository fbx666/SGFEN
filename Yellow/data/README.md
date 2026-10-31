# Yellow Data

> **Note:** The Yellow dataset used in this work is **not publicly released yet**.
> This folder is only a placeholder showing where the files should be placed. Once
> the dataset is available, a download link will be added here.

`SGFEN-Yellow.py` reads its inputs from this directory (paths are relative to the
dataset folder).

## Required files

| File | MATLAB variable | Description |
| --- | --- | --- |
| `9chengliu_HSI.mat` | `data` | Hyperspectral image, `H × W × B` |
| `9chengliu_lidar.mat` | `data` | LiDAR elevation map, `H × W` |
| `label.mat` | `label` | Ground truth, `H × W` (`0` = unlabeled); the script splits it into train/test |

The Yellow scene contains 9 land-cover classes. For each class, 20 samples are
randomly selected for training and the rest are used for testing (see
`class_sample_counts` in `utils.py`).

## How to place the files

If you have access to the dataset, put the three `.mat` files into this directory so
the layout becomes:

```
Yellow/
├── SGFEN-Yellow.py
├── utils.py
├── model/
└── data/
    ├── 9chengliu_HSI.mat
    ├── 9chengliu_lidar.mat
    └── label.mat
```

If your files use different names, either rename them as above or update the paths
returned by `selectdataset()` in `utils.py`.

Then run the script from the `Yellow/` folder:

```
cd Yellow
python SGFEN-Yellow.py
```
