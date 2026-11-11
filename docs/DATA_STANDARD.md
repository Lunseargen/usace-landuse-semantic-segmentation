# Data standard

Version 1 (2026-11-11). All stages of this repository obey these rules. The data registry is
[`data/REGISTRY.toml`](../data/REGISTRY.toml). Check it with:

```bash
python3 ~/.claude/skills/project-plan/scripts/registry_check.py data/REGISTRY.toml --plan PLAN.md
```

Change order: this file (and its version) → the registry → the ICD in `PLAN.md` section 5 →
`src/semseg/standard.py` → the tests. Task [0.2](../PLAN.md#t-0-2) puts each rule into code and
gives each rule a test.

## 1. Frame

| ID | Rule | Test |
|---|---|---|
| DS-1 | All rasters and vectors on disk use EPSG:5070 (NAD83 / CONUS Albers), unit metre. The web app can reproject for display only. | the CRS of each output file is EPSG:5070 |
| DS-2 | The 30 m grid is the Annual NLCD grid: origin x0 = −2415585, y0 = 3314805 (upper left), cell 30 m. This is the grid of the `usace-landuse` stack. | for each output raster, (xmin − x0) mod 30 = 0 and (y0 − ymax) mod 30 = 0; the transform of a stack raster gives the same lattice |
| DS-3 | A chunk is a 960 m square (32 × 32 cells) on the same origin. Its row is (y0 − ymax) / 960 and its column is (xmin − x0) / 960. | the chunk IDs of the code agree with `fine_grid_test/<pid>_<NAME>/chunks.csv` |
| DS-4 | The image grid is 0.5 m and nests in the 30 m grid: 60 × 60 image pixels in each cell, 1920 × 1920 in each chunk. | the NAIP chunk transform has a 0.5 m pixel and starts on a cell edge |
| DS-5 | The photo year is the NAIP year of each pixel (band `source_rank` and the file tags). A composition is valid only for its NAIP year. The history years are 1985–2025. | each cell record has `naip_year`; no composition has a year different from its NAIP year |

## 2. Identity and keys

| ID | Rule | Test |
|---|---|---|
| DS-6 | The project ID is the USACE `ombilSiteId` (integer). The project folder is `<pid>_<NAME>`, with the same name as the `usace-landuse` stack folder (`77_GRAPEVINE_LAKE`, `392_RAYSTOWN_LAKE`). | the folder names agree with the stack folder names |
| DS-7 | The chunk ID is `r<row>_c<col>` (DS-3). | the regular expression `^r\d+_c\d+$` |
| DS-8 | The cell row is (y0 − ymax) / 30 and the cell column is (xmin − x0) / 30. The cell ID is `row × 1 000 000 + col` (int64). Each table also has the columns `row` and `col`. | a round trip cell ID → bounds → cell ID |
| DS-9 | A polygon ID is `<chunk>_<run_id>_<n>`. A polygon ID is not stable from run to run. A join between runs uses the cell ID or the geometry, never the polygon ID. A label keeps its own geometry. | a label still finds its area after a new segmentation run |
| DS-10 | A run ID is `<YYYYMMDD-HHMM>-<method>` (for example `20261112-0930-sam2`). Each output file names its run ID. | the regular expression `^\d{8}-\d{4}-[a-z0-9_]+$` |

## 3. Codes

The class codes follow the NLCD legend (two digits, from the Anderson system: the first digit
is the level-1 class, the second digit is the level-2 class). Each NLCD code keeps its NLCD
meaning. The new photo classes use codes that NLCD does not use. Source: the NLCD legend of the
Annual NLCD Collection 1 (USGS, MRLC), and `usace-landuse` `scheme.py` for USACE-M.

<a id="codes"></a>

| ID | Rule |
|---|---|
| DS-11 | The code table below is the only class list. Code 0 = no photo (no data). Rasters use 0 for no data. |
| DS-12 | A labeler can give a level-1 code only (the tens digit × 10, for example 40 = "forest, type not known"). Level-1 codes are 10, 20, 30, 40, 50, 70, 80, 90. |
| DS-13 | The colour of each code is the official NLCD legend colour for an NLCD code, and a near shade for a new code (column "Colour"). These colours appear only in the map, the pie chart and the treemap. |

| L1 | Code | Name | NLCD code? | USACE-M (version 1) | Colour |
|---|---|---|---|---|---|
| 1 Water | 11 | Open water | yes | M01 | `#466b9f` |
| | 13 | River or stream | new | M01 (M25 if dry) | `#5a86b8` |
| | 14 | Pond | new | M01 | `#2f4f7a` |
| 2 Developed | 21 | Maintained grass (developed, open space) | yes | M10 + maintained flag | `#dec5c5` |
| | 25 | Building | new | M02 | `#ab0000` |
| | 26 | Paved road | new | M02 | `#eb0000` |
| | 27 | Unpaved road | new | M02 | `#c9846b` |
| | 28 | Parking or other paved | new | M02 | `#d99282` |
| | 29 | Dam or levee | new | M02 | `#7a3d3d` |
| 3 Barren | 31 | Barren, type not known | yes | M03 | `#b3ac9f` |
| | 32 | Sand or beach | new | M04 | `#e8dcb0` |
| | 33 | Rock or cliff | new | M05 | `#8f8a80` |
| | 34 | Mud flat or drawdown shore | new | M04 | `#a89b84` |
| | 35 | Gravel or quarry | new | M03 | `#c7c2b8` |
| 4 Forest | 41 | Deciduous forest | yes | M18–M24 by tree share (DS-15) | `#68ab5f` |
| | 42 | Evergreen forest | yes | M18–M24 by tree share | `#1c5f2c` |
| | 43 | Mixed forest | yes | M18–M24 by tree share | `#b5c58f` |
| | 44 | Forest, leaf type not known | new | M18, M19-21u, M22-24u | `#4f8a4a` |
| 5 Shrubland | 51 | Dwarf shrub | yes (Alaska) | M17 | `#af963c` |
| | 52 | Shrub or scrub | yes | M14-16u | `#ccb879` |
| 7 Herbaceous | 71 | Grassland or herbaceous | yes | M10 | `#dfdfc2` |
| 8 Planted or cultivated | 81 | Pasture or hay | yes | M12 | `#dcd939` |
| | 82 | Cultivated crops | yes | M13 | `#ab6c28` |
| | 83 | Fallow or bare field | new | M13 | `#c9a27a` |
| 9 Wetlands | 90 | Woody wetland | yes | M18–M24 by tree share + wetland flag | `#b8d9eb` |
| | 95 | Emergent herbaceous wetland | yes | M09 | `#6c9fb8` |
| | 96 | Aquatic bed | new | M08 | `#4f7f99` |

| ID | Rule | Test |
|---|---|---|
| DS-14 | The derived NLCD code of a cell (version 1): I = the share of codes 25–29. If I ≥ 0.20, the code is 22 (I < 0.50), 23 (I < 0.80) or 24. Else the code is the NLCD group with the largest share: 11, 13, 14 → 11; 21 → 21; 31–35 → 31; 41 → 41; 42 → 42; 43, 44 → 43; 51, 52 → 52; 71 → 71; 81 → 81; 82, 83 → 82; 90 → 90; 95, 96 → 95. | a unit test for each limit (0.199, 0.20, 0.50, 0.80) and each group |
| DS-15 | The USACE-M tree classes come from the tree share T (codes 41–44 and 90) of the cell, with the `usace-landuse` limits: T > 0.45 → closed (M22–M24), T ≥ 0.15 → open (M19–M21), T ≥ 0.10 → sparse (M18). The leaf type comes from the largest of 41, 42, 43; 44 alone → `u`. | a unit test for each limit |
| DS-16 | The USACE-M map is version 1. Task [0.2](../PLAN.md#t-0-2) checks it against `usace-landuse` `scheme.py` and records each change here. | the map has one row for each code |

## 4. Files

| ID | Rule | Test |
|---|---|---|
| DS-17 | Rasters: GeoTIFF, deflate, tiled 256 × 256, uint8 for codes and percent, float32 for other values; band descriptions set. | the profile of each output raster |
| DS-18 | Vectors: GeoPackage. Layer names: `polygons` (segmentation), `labels` (label export), `sample` (evaluation sample). | the layer names |
| DS-19 | Tables: Parquet (pyarrow) on the share; CSV only for small results in the repository (`results/`, less than 1 MB for each file). | the size of each file in `results/` |
| DS-20 | The cell record (one row for each cell, one file for each chunk, `cells/<chunk>.parquet`) has these columns: `cell_id`, `row`, `col`, `pid`, `chunk`, `naip_year`, `naip_date`, `photo_share` (the share of the cell with a photo), `share_<code>` (one column for each code, float, the sum over the codes is 1 ± 0.001 where `photo_share` > 0), `top_code`, `n_poly`, `nlcd_derived` (DS-14), `usace_m_derived` (DS-15), `nlcd_<year>`, `classf_<year>` and `tcc_<year>` for 1985–2025 (from the stack, read only), `run_id`. | the schema test of I4 |
| DS-21 | Each output file has this metadata (GeoTIFF tags, a GeoPackage metadata table, Parquet schema metadata): `run_id`, `code_version` (git commit), `data_standard` (version), `naip_year`, `sources` (the input files), `created` (ISO 8601 time). | the metadata test of each writer |
| DS-22 | Folder names on the share: `<root>/<pid>_<NAME>/{polygons,cells,rasters,labels,sample,sources}/`, `<root>/models/`, `<root>/runs/<run_id>/`. Do not name a folder `aux`, `con`, `nul` or `prn` (the SMB share changes these names). | a test of the folder-name function |

## 5. Storage

| ID | Rule | Test |
|---|---|---|
| DS-23 | Roots (see `[roots]` in the registry): `semseg` (the share folder of this project, read and write), `fgt` (`fine_grid_test`, read only), `stack` (the `usace-landuse` stack, read only). | the write function refuses a path under a read-only root |
| DS-24 | Large data (images, polygons, cell tables, rasters, model weights, caches) goes only under `semseg`. The repository keeps code, the configuration, the docs, and small results. | `git ls-files` has no file larger than 1 MB |
| DS-25 | A write to the share goes to a local temporary file first, then a copy to `<name>.part`, then `os.replace`. | the writer test |
| DS-26 | If a root does not exist (the share is not mounted), the step stops with exit code 3. It never writes the data in a different place. | the share-guard test |
| DS-27 | The app database is SQLite at `var/app.sqlite` on the local disk (SQLite is not safe on SMB). Each night, and on each admin request, the app exports the labels to `semseg/<pid>_<NAME>/labels/`. | the export test |
| DS-28 | Model weights and the Hugging Face cache are in `semseg/models/` (`HF_HOME`). | the set-up check prints `HF_HOME` |

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 1 | 2026-11-11 | first version, from the plan interview ([D5](../DECISIONS.md), [D6](../DECISIONS.md)) |
