# Houston Data

`SGFEN-Houston.py` reads its inputs from this directory (paths are relative to the
dataset folder), so the Houston dataset must be placed here before training.

## Required files

| File | MATLAB variable | Description |
| --- | --- | --- |
| `HSI.mat` | `HSI` | Hyperspectral image, `H × W × B` |
| `LiDAR.mat` | `LiDAR` | LiDAR / DSM elevation map, `H × W` |
| `TRLabel.mat` | `TRLabel` | Training ground truth, `H × W` (`0` = unlabeled) |
| `TSLabel.mat` | `TSLabel` | Test ground truth, `H × W` (`0` = unlabeled) |

The Houston scene contains 15 land-cover classes.

## Download

Baidu Netdisk: <https://pan.baidu.com/s/13c0h5A1fnJOKWkuG9oZlig?pwd=ouhw>
(password: `ouhw`)

## How to place the files

After extracting the archive, move the four `.mat` files into this directory so the
layout becomes:

```
Houston/
├── SGFEN-Houston.py
├── utils.py
├── model/
└── data/
    ├── HSI.mat
    ├── LiDAR.mat
    ├── TRLabel.mat
    └── TSLabel.mat
```

Then run the script from the `Houston/` folder:

```
cd Houston
python SGFEN-Houston.py
```
