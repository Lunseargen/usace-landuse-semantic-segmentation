# usace-landuse-semantic-segmentation

Semantic segmentation of NAIP 2022 imagery (0.5 m) for the USACE land-use research
(`~/workspace/usace-landuse`), with one record for each 30 m Annual NLCD cell (EPSG:5070).
Pilots: 392 Raystown Lake (PA) and 77 Grapevine Lake (TX). Prototype: samgeo (SAM 2 polygons,
text prompts, map vote). Later: a labeling and analysis web app, a blind random sample with
95 % intervals, and our own CNN.

- Plan: [`PLAN.md`](PLAN.md) (stages, tasks, requirements, risks, interfaces, FMEA).
- Decisions: [`DECISIONS.md`](DECISIONS.md).
- Data rules: [`docs/DATA_STANDARD.md`](docs/DATA_STANDARD.md) (grid, IDs, class codes, files, storage).
- Method reference: [`docs/METHOD_v1.md`](docs/METHOD_v1.md) (the `usace-landuse` method v1, quoted unchanged, 2026-11-11). It gives the USACE-M classes, the flags, the condition score and the change categories. It describes `usace_veg`, not our code (D3). Where it differs from the data standard, the data standard wins.
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
- `config.yaml`: the pipeline config. Keys: `share_root` (the data folder, see below), `aois` (the pilot AOI IDs), `segment` and `demo` (task 1.2).
- `semseg.standard`: the data standard in code (see below).
- `semseg._independence`: the scanner for D3 and R4. It finds imports of the `usace-landuse` code (`fine_grid`, `usace_veg`) in the source and in `pyproject.toml`.
- `tests/`: the pytest suite.

Dated notes (2026-11-11):

- `config.yaml` is the pipeline config. D15 covers only the registry (`data/REGISTRY.toml`).
- `load_config` refuses an empty `aois` list and duplicate keys. It raises `ConfigError`.

## Data standard in code

`semseg.standard` holds the data standard (`docs/DATA_STANDARD.md`, version 6). It has one pure
function or constant for each DS rule. It does no file access and no storage (task 0.3).

Main names:

- Grid: `cell_of_bounds`, `cell_id`, chunk IDs (`chunk_of_bounds`).
- Codes: `CODES` (with the text prompt), `PROMPTS`, `DERIVED_NLCD`, `colour`, `level1`.
- Labels (task 1.3): `vote_code`, `NLCD_VOTE`, `CLASSF_VOTE`, `check_machine_code`, `check_vote_code`.
- Derived codes: `derived_nlcd`, `derived_usace_m`, `usace_m_flags`, `top_code`.
- Change and condition (DS-31 to DS-34, D60): `change_group`, `change_category`, `condition_rating`,
  `condition_state`, `trend_slope`, `trajectory`, `score_change`, and the name tables `CHANGE_CATEGORIES`,
  `CONDITION_STATES`, `TRAJECTORIES`, `SCORE_CHANGES`. They are pure rules: no stage computes a stand,
  a score or a slope yet.

`tests/test_standard.py` has one test per DS rule. `test_grid_lattice` reads the real stack
rasters of 77 and 392 (read only). It fails, and never skips, when the share is missing.
It reads `usace_veg` `scheme.py` as text (D3).

Dated notes (2026-11-11):

- Grid math is exact (`int` or `Fraction`). Bounds off the 30 m lattice raise `ValueError`.
- Share sums round to 9 decimals before each limit (D34).
- D27 to D50 are `Draft (factory):` decisions. Jack reviews them.
- Version 6 (D60): 32 codes. New codes 12 (ice or snow, M03), 36 (boulder, cobble or talus, M06),
  53, 54, 55 (deciduous, evergreen, mixed shrub: M16, M15, M14). DS-15 sums the shrub codes 52 to 55
  into one shrubland target. `usace_m_flags(shares, photo_share, stack_flags=0)` copies the stack bits
  2 (F2 introduced) and 4 (F3 drawdown) from `flags_<naip_year>`. NLCD 12 now votes for code 12.

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
- The files of 2026-11-11 have `code_version` `6df2dd9c648ec151830e48c29d7e8df0ced66fdc` (commit "1.1 Chunk grid").

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

## SAM 2 polygons (`semseg.segment`, task 1.2, 2026-11-11)

`semseg.segment` makes polygons from NAIP chunks with SAM 2. The model sits behind a `MaskModel` interface. The `samgeo` adapter is one implementation. The tests use a fake model.

Warning: Do not run the command until Jack says run (D22). It loads the real model and downloads the weights.

Command (only after Jack says run):

```
uv run python -m semseg.segment --demo --run
uv run python -m semseg.segment --pid 77 --run
```

- `--demo` runs the demo chunks of `config.yaml`. `--pid <pid>` runs all chunks of `<semseg>/<pid>_<NAME>/chunks.parquet`, with one run log for each pilot (D59, 2026-11-11). Give one of the 2 options, not both.
- The `--run` flag is required. Abbreviations such as `--ru` are refused with exit code 2. Without `--run`, the command loads nothing.
- Before any model load, the code calls `ensure_hf_home()` (D38) and reads the available memory (`free -m`). If the value is below `segment.min_free_mb` (16000 MB in `config.yaml`), it exits with code 4 and loads nothing (D47, 2026-11-11).
- The floor is 8000 MB (`MIN_FREE_FLOOR_MB`). A lower `min_free_mb` is a config error (D50, 2026-11-11).
- A run refuses to start when `<semseg>/runs/<run_id>/` exists (D49, 2026-11-11). Wait one minute, or use another method suffix, for a new run ID.
- Outputs:
  - `<semseg>/<pid>_<NAME>/polygons/<chunk>.gpkg`, layer `polygons`. The IDs follow DS-9. The layer metadata follows DS-21.
  - `<semseg>/runs/<run_id>/log.json`. It has the time, the memory and GPU state before and after, the `fake` flag and `log_errors`.
- The demo uses 6 chunks of pilot 77. The list is `demo.chunks` in `config.yaml` (D46, 2026-11-11).
- The Hugging Face hub writes the SAM 2 weights to `HF_HOME/hub` on the share. This is the only write that does not use `safe_write` (D48, 2026-11-11; data standard version 4).
- The share refuses `FileLock` and `fchmod` (errno 95). Before the real load, `storage.use_soft_hf_locks()` makes the hub use `SoftFileLock` (D58, 2026-11-11). If a run stops in a download, remove the 0-byte lock file in `<semseg>/models/hub/.locks/` before the next run.
- The test suite never loads the real model. `tests/conftest.py` blocks the model classes and the Hugging Face download functions (FM-71), the LangSAM, GroundingDINO, `torch.hub` and `transformers` loaders (D55, FM-89), and sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` to 1.
- NAIP source (D57, 2026-11-11, data standard version 5): `read_chunk` refuses a chunk file whose tag `gsd` is not `0.6` (`ValueError`). `chunk_naip_year(path)` returns the `years` tag (one year only). `run_chunks` writes that year into the `naip_year` key of the polygon file. The chunk files stay on the 0.5 m grid.

## Polygon labels (`semseg.label`, task 1.3, 2026-11-11)

`semseg.label` adds 4 columns to `polygons/<chunk>.gpkg`: `code_prompt`, `p_prompt` (LangSAM text prompts, DS-30) and `code_vote`, `p_vote` (map vote of `nlcd_2022` and `classf_2022`, DS-29). The prompt of each code is in the code table (DS-11, `standard.PROMPTS`). The settings are in the `label:` section of `config.yaml` (D51, D53).

Warning: Do not run the command until Jack says run (D22). It loads LangSAM (GroundingDINO and SAM 2) on the GPU.

Command (only after Jack says run):

```
uv run python -m semseg.label --demo --run
```

- The command follows `semseg.segment`: `--run` is required, the memory check (exit code 4) comes before the model load, and a run refuses an existing `runs/<run_id>/` (the label run ID has the method `langsam`).
- The vote area-weights the 30 m pixels. A `code_vote` of 20 means "Developed" at level 1 (D56). A machine label is 0 or one of the 32 codes (D54, D60).
- The run log `<semseg>/runs/<run_id>/log.json` has the key `codes`: 34 rows (code 0, code 20, the 32 codes) with the polygon count and area. A code with 0 polygons is a row with zeros (FM-2). Task 1.5 writes `results/demo/codes.csv`.
- The code is done with fake-model tests. The real run, and the quality of the 32 prompts, wait for Jack.
