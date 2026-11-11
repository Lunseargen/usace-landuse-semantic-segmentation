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
- `semseg._independence`: the scanner for D3 and R4. It finds imports of the `usace-landuse` code (`fine_grid`, `usace_veg`) in the source and in `pyproject.toml`.
- `tests/`: the pytest suite.

Dated notes (2026-11-11):

- `config.yaml` is the pipeline config. D15 covers only the registry (`data/REGISTRY.toml`).
- `load_config` refuses an empty `aois` list and duplicate keys. It raises `ConfigError`.

## Where the data is

Large data is on the lab share, not in this repository:

- This project (read and write): `/run/user/1001/gvfs/smb-share:server=100.102.218.98,share=lab/data/usace-landuse-semseg/`
- NAIP and sources from `fine_grid` (read only): `.../lab/data/usace-landuse/fine_grid_test/`
- The `usace-landuse` 30 m stack (read only): `~/workspace/usace-landuse/USACE/01_Data/stack/`

The share is a GVFS mount. It exists only while the user is logged in on the desktop. If it
is missing, the steps stop (exit code 3). They never write the data to the local disk.
