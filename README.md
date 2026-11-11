# usace-landuse-semantic-segmentation

Semantic segmentation of NAIP 2022 imagery (0.5 m) for the USACE land-use research
(`~/workspace/usace-landuse`), with one record for each 30 m Annual NLCD cell (EPSG:5070).
Pilots: 392 Raystown Lake (PA) and 77 Grapevine Lake (TX). Prototype: samgeo (SAM 2 polygons,
text prompts, map vote). Later: a labeling and analysis web app, a blind random sample with
95 % intervals, and our own CNN.

- Plan: [`PLAN.md`](PLAN.md) (stages, tasks, requirements, risks, interfaces, FMEA).
- Decisions: [`DECISIONS.md`](DECISIONS.md).
- Data rules: [`docs/DATA_STANDARD.md`](docs/DATA_STANDARD.md) (grid, IDs, class codes, files, storage).
- Data locations: [`data/REGISTRY.toml`](data/REGISTRY.toml). Check it from the repository root:
  `python3 ~/.claude/skills/project-plan/scripts/registry_check.py data/REGISTRY.toml --plan PLAN.md`

## Set-up

Run these commands from the repository root.

Install the environment:

```
uv sync --python 3.12
```

Run the tests with the coverage gate (100 % of `src/semseg`):

```
uv run pytest --cov=semseg --cov-fail-under=100
```

Run the task gate:

```
python3 ~/.claude/skills/build-task/scripts/gate.py
```

Resolved stack (from `uv.lock`):

- Python 3.12, NVIDIA GB10 (aarch64).
- torch 2.14.1+cu130 and torchvision 0.29.1+cu130 (CUDA 13 wheels, index `pytorch-cu130`).
- segment-geospatial 1.4.2 (module `samgeo`), extras `samgeo2` and `text`.

Package layout:

- `src/semseg/`: the package. `semseg.config.load_config` reads `config.yaml`.
- `config.yaml`: the pipeline config. Keys: `share_root` (the data folder, see below) and `aois` (the pilot AOI IDs).
- `semseg.standard`: the data standard in code (see below).
- `semseg._independence`: the scanner for D3 and R4. It finds imports of the `usace-landuse` code (`fine_grid`, `usace_veg`) in the source and in `pyproject.toml`.
- `tests/`: the pytest suite.

Dated notes (2026-11-11):

- `config.yaml` is the pipeline config. D15 covers only the registry (`data/REGISTRY.toml`).
- `load_config` refuses an empty `aois` list and duplicate keys. It raises `ConfigError`.

## Data standard in code

`semseg.standard` holds the data standard (`docs/DATA_STANDARD.md`, version 3). It has one pure
function or constant for each DS rule. It does no file access and no storage (task 0.3).

Main names:

- Grid: `cell_of_bounds`, `cell_id`, chunk IDs (`chunk_of_bounds`).
- Codes: `CODES`, `DERIVED_NLCD`, `colour`, `level1`.
- Derived codes: `derived_nlcd`, `derived_usace_m`, `usace_m_flags`, `top_code`.

`tests/test_standard.py` has one test per DS rule. `test_grid_lattice` reads the real stack
rasters of 77 and 392 (read only). It fails, and never skips, when the share is missing.
It reads `usace_veg` `scheme.py` as text (D3).

Dated notes (2026-11-11):

- Grid math is exact (`int` or `Fraction`). Bounds off the 30 m lattice raise `ValueError`.
- Share sums round to 9 decimals before each limit (D34).
- D27 to D45 are `Draft (factory):` decisions. Jack reviews them.

## Chunk grid

`semseg.grid` builds the chunk list of each pilot (task 1.1). It writes
`<semseg>/<pid>_<NAME>/chunks.parquet` through `semseg.storage.safe_write`. The schema is in
`docs/DATA_STANDARD.md` DS-22 and `PLAN.md` ICD I9.

Command:

```
uv run python -m semseg.grid 77 392
```

Dated notes (2026-11-11):

- Rule (D43, D44): a chunk is in the list when its 960 m box shares an area > 0 with the footprint
  (`boundary.gpkg`) buffered by 150 m. A chunk that only touches the buffer edge is not in the list.
- Result: 157 chunks for 77 and 227 for 392. Both ID sets equal `chunks.csv` of `fine_grid_test`.
  The CLI does not read `chunks.csv`. `test_chunks_match_fine_grid` does (read only).
- All pids are checked before the first write. A bad pid or a bad frame writes nothing.
- DS-21 `code_version` (D45): the full commit SHA, with `-dirty` when tracked files have changes.
  Without git or a commit, the command raises an error and writes nothing.
- The files of 2026-11-11 have `-dirty` (uncommitted work). Run the command again after the commit.

## Where the data is

Large data is on the lab share, not in this repository:

- This project (read and write): `/run/user/1001/gvfs/smb-share:server=100.102.218.98,share=lab/data/usace-landuse-semseg/`
- NAIP and sources from `fine_grid` (read only): `.../lab/data/usace-landuse/fine_grid_test/`
- The `usace-landuse` 30 m stack (read only): `~/workspace/usace-landuse/USACE/01_Data/stack/`

The share is a GVFS mount. It exists only while the user is logged in on the desktop. If it
is missing, the steps stop (exit code 3). They never write the data to the local disk.

## Storage and set-up check

Storage (`semseg.storage`, 2026-11-11):

- The roots come from `data/REGISTRY.toml` `[roots]`. The `share_root` value in `config.yaml` must equal the `semseg` root (D36).
- The share guard exits with code 3 when the share is missing. It never creates the share path.
- The `fgt` and `stack` roots are read only. A write under them raises an error.
- A write goes to a unique part file in the target folder, `<name>.<8 hex>.part`. Then `os.replace` moves it to `<name>` (D39).
- `ensure_hf_home()` sets `HF_HOME` to `<semseg>/models` when it is not set (D38). If `HF_HOME` is set outside the semseg root, it raises `ValueError`. Call it before any Hugging Face import.

Set-up check (`semseg.setup_check`, 2026-11-11):

- Run it from the repository root: `uv run python -m semseg.setup_check`
- It reads the share, the stack and the GPU state. It writes nothing to the share or the stack, and it starts no GPU run.
- Each run writes `results/setup_check.json`. A failed check sets `pass` to `false` and gives the `reason` (D40).
- Exit codes: 0 = pass, 1 = other failure, 3 = share missing.
- Last result (2026-11-11): `pass` true. NAIP 2022 chunks: 157 for 77 and 227 for 392, total 384 (D35).
