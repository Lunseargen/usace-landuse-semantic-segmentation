# Data standard

Version 3 (2026-11-11). All stages of this repository obey these rules. The data registry is
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
| DS-3 | A chunk is a 960 m square (32 × 32 cells) on the same origin. Its row is floor((y0 − ymax) / 960) and its column is floor((xmin − x0) / 960). The chunk of a cell is (floor(row / 32), floor(col / 32)). A row or a column below 0 is outside the grid and is an error. The chunk list of a pilot has each chunk whose box has a positive area inside the footprint (`boundary.gpkg`) buffered by 150 m (D13). A chunk that only touches the edge of the buffer is not in the list (D44). | the chunk IDs of the code agree with `fine_grid_test/<pid>_<NAME>/chunks.csv`; the chunk list of 77 has 157 chunks and the list of 392 has 227; the ID sets agree with `chunks.csv` |
| DS-4 | The image grid is 0.5 m and nests in the 30 m grid: 60 × 60 image pixels in each cell, 1920 × 1920 in each chunk. In a chunk, the pixel of a point is i = floor((ymax − y) / 0.5), j = floor((x − xmin) / 0.5); its cell in the chunk is (floor(i / 60), floor(j / 60)). | the NAIP chunk transform has a 0.5 m pixel and starts on a cell edge |
| DS-5 | The photo year is the NAIP year of each pixel (band `source_rank` and the file tags). A composition is valid only for its NAIP year. The history years are 1985–2025. | each cell record has `naip_year`; no composition has a year different from its NAIP year |

## 2. Identity and keys

| ID | Rule | Test |
|---|---|---|
| DS-6 | The project ID is the USACE `ombilSiteId` (integer). The project folder is `<pid>_<NAME>`, with the same name as the `usace-landuse` stack folder (`77_GRAPEVINE_LAKE`, `392_RAYSTOWN_LAKE`). In `data/REGISTRY.toml`, the placeholder `{pid}` is the project folder name `<pid>_<NAME>`. The column `pid` of a table is the integer. | the folder names agree with the stack folder names |
| DS-7 | The chunk ID is `r<row>_c<col>` (DS-3). | the regular expression `^r\d+_c\d+$` |
| DS-8 | The cell row is (y0 − ymax) / 30 and the cell column is (xmin − x0) / 30. Both must be whole numbers (exact, no tolerance); else the bounds are off the grid and the step stops with an error. A point (x, y) is in the cell row floor((y0 − y) / 30), column floor((x − x0) / 30): a cell includes its left edge and its top edge. The cell ID is `row × 1 000 000 + col` (int64), with row ≥ 0 and 0 ≤ col < 1 000 000. Each table also has the columns `row` and `col`. | a round trip cell ID → bounds → cell ID; a point off the lattice raises an error |
| DS-9 | A polygon ID is `<chunk>_<run_id>_<n>`. A polygon ID is not stable from run to run. A join between runs uses the cell ID or the geometry, never the polygon ID. A label keeps its own geometry. | a label still finds its area after a new segmentation run |
| DS-10 | A run ID is `<YYYYMMDD-HHMM>-<method>` (for example `20261112-0930-sam2`). Each output file names its run ID. | the regular expression `\A[0-9]{8}-[0-9]{4}-[a-z0-9_]+\Z` (full match, ASCII digits only; the date and time must be real, checked with `strptime`) |

## 3. Codes

The class codes follow the NLCD legend (two digits, from the Anderson system: the first digit
is the level-1 class, the second digit is the level-2 class). Each NLCD code keeps its NLCD
meaning. The new photo classes use codes that NLCD does not use. Source: the NLCD legend of the
Annual NLCD Collection 1 (USGS, MRLC), and `usace-landuse` `scheme.py` for USACE-M.

<a id="codes"></a>

| ID | Rule |
|---|---|
| DS-11 | The code table below is the only class list for the photo classes and the labels (27 codes). Code 0 = no photo (no data). Rasters and cell records use 0 for no data. The derived NLCD code (DS-14) uses the NLCD legend codes of DS-14; they are not photo classes. A stack NLCD value of 250 (no data) becomes 0 in a cell record. |
| DS-12 | A labeler can give a level-1 code only. The level-1 code is the tens digit × 10 (for example 40 = "forest, type not known"), except for wetlands: 99 = "wetland, type not known", because NLCD code 90 means woody wetland (D6). Level-1 codes are 10, 20, 30, 40, 50, 70, 80, 99. A label can have a level-1 code; a composition (DS-14, DS-15, DS-20) cannot. |
| DS-13 | The colour of each code is the official NLCD legend colour for an NLCD code, and a near shade for a new code (column "Colour"). These colours appear only in the map, the pie chart and the treemap. Each colour is different from all other colours of the code table and of the derived NLCD codes 22, 23, 24. A new code never uses an NLCD legend colour. The level-1 codes have no colour in version 2. |

| L1 | Code | Name | NLCD code? | USACE-M (version 1) | Colour |
|---|---|---|---|---|---|
| 1 Water | 11 | Open water | yes | M01 | `#466b9f` |
| | 13 | River or stream | new | M01 | `#5a86b8` |
| | 14 | Pond | new | M01 | `#2f4f7a` |
| 2 Developed | 21 | Developed, open space (maintained grass) | yes | M10 + maintained flag | `#dec5c5` |
| | 25 | Building | new | M02 | `#8b0000` |
| | 26 | Paved road | new | M02 | `#ff3b3b` |
| | 27 | Unpaved road | new | M02 | `#c9846b` |
| | 28 | Parking or other paved | new | M02 | `#e3aa9c` |
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
| 9 Wetlands (level-1 code 99) | 90 | Woody wetland | yes | M18–M24 by tree share + wetland flag | `#b8d9eb` |
| | 95 | Emergent herbaceous wetland | yes | M09 | `#6c9fb8` |
| | 96 | Aquatic bed | new | M08 | `#4f7f99` |

| ID | Rule | Test |
|---|---|---|
| DS-14 | The derived NLCD code of a cell (version 2). A cell with photo_share = 0 has code 0. I = the share of codes 25–29. If I ≥ 0.20, the code is 22 (I < 0.50), 23 (0.50 ≤ I < 0.80) or 24 (I ≥ 0.80). Else the code is the NLCD group with the largest share: 11, 13, 14 → 11; 21 → 21; 31–35 → 31; 41 → 41; 42 → 42; 43, 44 → 43; 51, 52 → 52; 71 → 71; 81 → 81; 82, 83 → 82; 90 → 90; 95, 96 → 95. On a tie, the lowest group code. Comparisons are exact (no tolerance) on group sums that are rounded to 9 decimals first (D34). The output codes have the NLCD name and the NLCD legend colour: 22 `#d99282`, 23 `#eb0000`, 24 `#ab0000`; the other codes as in the code table. | a unit test for each limit (0.199, 0.20, 0.499, 0.50, 0.799, 0.80), each group and a tie |
| DS-15 | The USACE-M code of a cell (usace_m_derived, version 2). A cell with photo_share = 0 has code 0. If I (DS-14) ≥ 0.20, the code is M02, before the tree classes (as usace-landuse rule 5). Else T = the share of codes 41–44 and 90. T > 0.45 → closed tree canopy; 0.15 ≤ T ≤ 0.45 → open tree canopy; 0.10 ≤ T < 0.15 → sparse tree canopy (M18). The leaf type of a closed or open cell comes from the largest share of 41, 42, 43: 41 → deciduous (M24, M21), 42 → evergreen (M23, M20), 43 → mixed (M22, M19); a tie of two or three → mixed. 44 and 90 have no leaf type: if 41, 42 and 43 are all 0, the code is M22-24u or M19-21u. If T < 0.10, the code is the USACE-M code (DS-16) with the largest summed share of the non-tree codes; on a tie, the lowest raster code. The limits 45, 15 and 10 are the usace-landuse classify values (tcc_closed, tcc_open, tcc_sparse, release 2026-10-31). There they apply to the USFS TCC canopy percent; here they apply to the photo tree share T, a different measure. Comparisons are exact (no tolerance) on group sums that are rounded to 9 decimals first (D34). | a unit test for each limit (I 0.199 and 0.20; T 0.099, 0.10, 0.149, 0.15, 0.45, 0.451), each leaf case and each non-tree code |
| DS-16 | The USACE-M map is version 1 (column "USACE-M" of the code table). Task [0.2](../PLAN.md#t-0-2) checked it against usace-landuse `scheme.py` (`CLASSES`) on 2026-11-11. A raster or a cell record stores the `scheme.py` raster code (uint8), not the label: M01–M24 → 1–24, M04-06u → 25, M14-16u → 26, M19-21u → 27, M22-24u → 28, M25 → 29; 0 = no data. Code 13 maps to M01: M25 (dry channel) needs a water time series, so the photo map does not give it. Codes 32 and 34 map to M04, as usace-landuse rules 7d (beach) and 7f (drawdown shore). The photo map does not give M04-06u, M06, M07, M11, M14, M15, M16 or M25; they stay valid in classf_<year>. Flags (usace_m_flags, bits as usace-landuse flags_<year>.tif): 1 = woody wetland (F1), if the code is a tree class (M18–M24, M19-21u, M22-24u) and the share of 90 ≥ the share of 41–44; 128 = maintained grass (new bit; usace-landuse uses bits 1–64), if the code is M10 and the share of 21 ≥ the share of 71. | each label and raster code agrees with `scheme.py`; a unit test for each flag |

## 4. Files

| ID | Rule | Test |
|---|---|---|
| DS-17 | Rasters: GeoTIFF, deflate, tiled 256 × 256, uint8 for codes and percent, float32 for other values; band descriptions set. | the profile of each output raster |
| DS-18 | Vectors: GeoPackage. Layer names: `polygons` (segmentation), `labels` (label export), `sample` (evaluation sample). | the layer names |
| DS-19 | Tables: Parquet (pyarrow) on the share; CSV only for small results in the repository (`results/`, less than 1 MB for each file). | the size of each file in `results/` |
| DS-20 | The cell record (one row for each cell, one file for each chunk, `cells/<chunk>.parquet`) has these columns, in this order: `cell_id`, `row`, `col`, `pid`, `chunk`, `naip_year`, `naip_date`, `photo_share` (the share of the cell with a photo), `share_<code>` (one float column for each of the 27 codes of the code table, in table order, as a share of the photo area; no column for code 0, the level-1 codes or 22–24; where `photo_share` > 0 the sum over the codes, rounded to 9 decimals (D34), is 1 ± 0.001, both ends included; where `photo_share` = 0 each share is 0), `top_code` (the code with the largest share; on a tie the lowest code; 0 if `photo_share` = 0), `n_poly`, `nlcd_derived` (DS-14), `usace_m_derived` (DS-15, a raster code of DS-16), `usace_m_flags` (DS-16), `nlcd_<year>` (NLCD 250 → 0, DS-11), `classf_<year>` and `tcc_<year>` for 1985–2025 (from the stack, read only; first all nlcd years, then all classf years, then all tcc years), `run_id`. | the schema test of I4 |
| DS-21 | Each output file has this metadata (GeoTIFF tags, a GeoPackage metadata table, Parquet schema metadata): `run_id`, `code_version` (git commit), `data_standard` (version), `naip_year`, `sources` (the input files), `created` (ISO 8601 time, UTC with the label `Z`). `code_version` is the full SHA of `git rev-parse HEAD`, plus `-dirty` when `git status --porcelain --untracked-files=no` is not empty; a writer that cannot read the commit stops and writes nothing (D45). | the metadata test of each writer |
| DS-22 | Folder names on the share: `<root>/<pid>_<NAME>/{polygons,cells,rasters,labels,sample,sources,baselines}/`, the chunk list `<root>/<pid>_<NAME>/chunks.parquet`, `<root>/models/`, `<root>/runs/<run_id>/`. Do not name a folder or a file `aux`, `con`, `nul` or `prn`, with or without an extension, in any letter case (the SMB share changes these names). `chunks.parquet` has these columns in this order: `chunk` (string, DS-7), `row`, `col` (int64, DS-3), `xmin`, `ymin`, `xmax`, `ymax` (float64, EPSG:5070, the chunk box), `footprint_share` (float64: the area of the chunk inside the footprint without the buffer, divided by 960², rounded to 4 decimals), `pid` (int64, DS-6). The rows are sorted by `row`, then `col`; `chunk` is unique. The file has the DS-21 metadata in the Parquet key-value metadata. The writer is `safe_write` (DS-25) (D43). | a test of the folder-name function; the schema test and the metadata test of `semseg.grid` |

## 5. Storage

| ID | Rule | Test |
|---|---|---|
| DS-23 | Roots (see `[roots]` in the registry): `semseg` (the share folder of this project, read and write), `fgt` (`fine_grid_test`, read only), `stack` (the `usace-landuse` stack, read only). | the write function refuses a path under a read-only root |
| DS-24 | Large data (images, polygons, cell tables, rasters, model weights, caches) goes only under `semseg`. The repository keeps code, the configuration, the docs, and small results. | `git ls-files` has no file larger than 1 MB |
| DS-25 | A write to the share goes to a local temporary file first, then a copy to a new file `<name>.<8 hex>.part` in the target folder (unique name, created with `O_EXCL` and `O_NOFOLLOW`), then `os.replace`. The code removes its own part file on an error and checks the share again right before the replace. | the writer test |
| DS-26 | If a root does not exist (the share is not mounted), the step stops with exit code 3. It never writes the data in a different place. | the share-guard test |
| DS-27 | The app database is SQLite at `var/app.sqlite` on the local disk (SQLite is not safe on SMB). Each night, and on each admin request, the app exports the labels to `semseg/<pid>_<NAME>/labels/`. | the export test |
| DS-28 | Model weights and the Hugging Face cache are in `semseg/models/` (`HF_HOME`). | the set-up check prints `HF_HOME` |

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 1 | 2026-11-11 | first version, from the plan interview ([D5](../DECISIONS.md), [D6](../DECISIONS.md)) |
| 2 | 2026-11-11 | Draft (factory), task 0.2: level-1 wetland code 99; derived NLCD list; near shades for 25, 26, 28; name of 21; NLCD 250 → 0 ([D27](../DECISIONS.md)). DS-14 and DS-15 version 2: no photo, ties, leaf type, T < 0.10, exact limits ([D28](../DECISIONS.md)); in DS-15, I ≥ 0.20 → M02 before the tree classes ([D33](../DECISIONS.md)). DS-16 raster codes and flags; new column usace_m_flags ([D29](../DECISIONS.md)). Floor and exact lattice check in DS-3, DS-4, DS-8 ([D30](../DECISIONS.md)). DS-6 registry {pid}; DS-22 baselines and chunks.parquet ([D31](../DECISIONS.md)). DS-20 share columns and top_code ([D32](../DECISIONS.md)). Fix round: DS-10 pattern is `\A[0-9]{8}-[0-9]{4}-[a-z0-9_]+\Z` for a full match, with a real date and time; DS-14, DS-15 and DS-20 round each group share sum to 9 decimals ([D34](../DECISIONS.md)). DS-25 unique part name `<name>.<8 hex>.part`, `O_EXCL`, `O_NOFOLLOW`, guard again before the replace ([D39](../DECISIONS.md)). |
| 3 | 2026-11-11 | Draft (factory), task 1.1: chunk list rule in DS-3 (positive area inside the footprint buffered by 150 m; an edge touch is out) ([D44](../DECISIONS.md)); `chunks.parquet` schema in DS-22 ([D43](../DECISIONS.md)). DS-21: `code_version` is the full SHA plus `-dirty`; the writer stops when git fails ([D45](../DECISIONS.md)). |
