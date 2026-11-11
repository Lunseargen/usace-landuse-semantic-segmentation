# CLAUDE.md

Semantic segmentation of NAIP 0.5 m imagery, summarised on the Annual NLCD 30 m grid, for the
USACE land-use research. Read [`PLAN.md`](PLAN.md) first: its status line names the next task.
Decisions are in [`DECISIONS.md`](DECISIONS.md).

## Commands (2026-11-11)

- Set up the environment: `uv sync --python 3.12`.
- Run the tests with the coverage gate: `uv run pytest --cov=semseg --cov-fail-under=100`.
- Run the task gate: `python3 ~/.claude/skills/build-task/scripts/gate.py`.
- Write the chunk list of each pilot (task 1.1): `uv run python -m semseg.grid <pid> ...`.
- Run the set-up check (read only, no GPU run): `uv run python -m semseg.setup_check`. Exit code 3 = the lab share is not mounted.
- Run the SAM 2 polygons (task 1.2) (only after Jack says run): `uv run python -m semseg.segment --demo --run`.

## Rules (2026-11-11)

- Obey [`docs/DATA_STANDARD.md`](docs/DATA_STANDARD.md). To change a rule: the standard (and its
  version) first, then `data/REGISTRY.toml`, the ICD in `PLAN.md` section 5, the code, the tests.
- Each data set has an entry in `data/REGISTRY.toml`. Run
  `python3 ~/.claude/skills/project-plan/scripts/registry_check.py data/REGISTRY.toml --plan PLAN.md`
  after a data change. Exit code 3 = the lab share is not mounted: stop and tell the user.
- The code is independent: do not import or copy code from `usace-landuse` (`fine_grid`,
  `usace_veg`). Their data is read only (D3).
- Large data goes only to `<share>/data/usace-landuse-semseg/` (D23). Never write under
  `fine_grid_test` or the stack.
- Write data only through `semseg.storage` (share guard, read-only roots, safe write).
- Do not start a GPU run (SAM 2, LangSAM, training) until the user says so (D22). The vLLM
  server holds about 97 GB of the 121 GB GB10 memory.
- Do not run `python -m semseg.segment --run` until Jack says so (D22).
- The web app uses the `vllmPortal` design (`~/workspace/vllmPortal/DESIGN.md`) and the login
  pattern of `USDA-LFD-LLM` `scripts/scrape/queue_server.py` (D16). No CDN: vendor each library.
