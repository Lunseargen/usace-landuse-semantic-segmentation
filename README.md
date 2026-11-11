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

## Where the data is

Large data is on the lab share, not in this repository:

- This project (read and write): `/run/user/1001/gvfs/smb-share:server=100.102.218.98,share=lab/data/usace-landuse-semseg/`
- NAIP and sources from `fine_grid` (read only): `.../lab/data/usace-landuse/fine_grid_test/`
- The `usace-landuse` 30 m stack (read only): `~/workspace/usace-landuse/USACE/01_Data/stack/`

The share is a GVFS mount. It exists only while the user is logged in on the desktop. If it
is missing, the steps stop (exit code 3). They never write the data to the local disk.
