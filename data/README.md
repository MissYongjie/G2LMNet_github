# Data preparation

G2LMNet is evaluated on six public heterogeneous change-detection pairs (HetData #1–#6).
Each pair is a MATLAB `.mat` file with three variables:

| Variable   | Shape       | Description                                   |
|------------|-------------|-----------------------------------------------|
| `image_t1` | H × W × C   | pre-event image (uint8 or float)              |
| `image_t2` | H × W × C   | post-event image, same grid and channel count |
| `Ref_gt`   | H × W       | reference change map (non-zero = changed)     |

`Ref_gt` is used **only** for evaluation and logging, never for training.

## Download

Download `HetData_G2LMNet.zip` from the [Releases](../../releases) page and unzip it here:

```
data/
└── hetdata/
    ├── hetdata1_gloucester1.mat
    ├── hetdata2_texas.mat
    ├── hetdata3_gloucester2.mat
    ├── hetdata4_sardinia.mat
    ├── hetdata5_toulouse.mat
    └── hetdata6_shuguang.mat
```

These are the exact arrays used in the paper (three-channel versions of every pair), repackaged
with only the three variables above. Files with the older names used by some public
releases (`dataset#2.mat`, `TexasALI.mat`, `dataset#6.mat`, `dataset#1.mat`, `dataset#3.mat`,
`dataset#4.mat`, in the order of the table below) are also recognised.

| ID | Name          | Sensors (t1 / t2)          | Size (H × W) | Event                 | GSD    |
|----|---------------|----------------------------|--------------|-----------------------|--------|
| #1 | Gloucester-I  | SPOT / NDVI                | 990 × 554    | flooding              | ≈25 m  |
| #2 | Texas         | Landsat-5 / EO-1 ALI       | 1534 × 808   | forest fire           | 30 m   |
| #3 | Gloucester-II | QuickBird-2 / TerraSAR-X   | 1034 × 582   | flooding              | 0.65 m |
| #4 | Sardinia      | Landsat-5 / Google Earth   | 300 × 412    | lake expansion        | 30 m   |
| #5 | Toulouse      | Pléiades / WorldView-2     | 500 × 500    | construction          | 0.52 m |
| #6 | Shuguang      | Radarsat-2 / Google Earth  | 593 × 921    | building construction | 8 m    |

All credit for the data goes to the original providers; please cite the original dataset
papers when you use them.

## Using your own pair

Save the co-registered images as `image_t1` / `image_t2` (same H, W and C; replicate or
reduce bands if needed) and a `Ref_gt` (zeros if unknown), copy any config in
`configs/` and change `dataset.file`. Use `t1_norm/t2_norm: minmax` for two similar optical
images, or `zscore` / `log_zscore` (log-transform for speckled or index-like data) otherwise.
