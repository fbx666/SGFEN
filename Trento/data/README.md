# Trento Data

`SGFEN-Trento.py` reads its inputs from this directory (paths are relative to the
dataset folder), so the Trento dataset must be placed here before training.

## Required files

| File | MATLAB variable | Description |
| --- | --- | --- |
| `HSI.mat` | `HSI` | Hyperspectral image, `H × W × B` |
| `LiDAR.mat` | `LiDAR` | LiDAR elevation map, `H × W` |
| `TRLabel.mat` | `TRLabel` | Training ground truth, `H × W` (`0` = unlabeled) |
| `TSLabel.mat` | `TSLabel` | Test ground truth, `H × W` (`0` = unlabeled) |

The Trento scene contains 6 land-cover classes.

## Download

Baidu Netdisk: <https://pan.baidu.com/s/1fTM4iUZOdd7LeZ1iKkVvoA?pwd=6z62>
(password: `6z62`)

## How to place the files

After extracting the archive, move the four `.mat` files into this directory so the
layout becomes:

```
Trento/
├── SGFEN-Trento.py
├── utils.py
├── model/
└── data/
    ├── HSI.mat
    ├── LiDAR.mat
    ├── TRLabel.mat
    └── TSLabel.mat
```

Then run the script from the `Trento/` folder:

```
cd Trento
python SGFEN-Trento.py
```
