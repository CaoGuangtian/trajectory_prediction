# trajectory_prediction project structure notes

This file is for Codex/agent working context. It is not the user-facing project documentation.
User-facing design docs should go under `docs/` when needed.

## Project Positioning

`trajectory_prediction` is an offline trajectory prediction experiment platform.

It should handle:

- multiple datasets
- multiple preprocessing methods and configurations
- multiple prediction models and configurations
- quantitative evaluation
- offline/web-based result visualization

It is not the real-time trajectory system. Do not mix it with `RealTimeTrajectory`.

## Major Boundaries

The project should be organized around the data flow:

```text
raw data
  -> dataset reader
  -> canonical schema view/cache
  -> preprocessing artifact
  -> sample artifact
  -> training / inference
  -> prediction artifact
  -> evaluation / visualization
```

The most important stable boundaries are:

- canonical schema
- preprocessed artifact
- sample artifact
- prediction artifact
- experiment manifest

Preprocessing methods and prediction models should not call each other directly.
They should connect through saved artifacts and manifests.

## Data Directory

Use a dataset-centered top-level structure:

```text
data/
  <dataset_name>/
    <source files or internal subsets>/
```

Each direct child of `data/` is one dataset. Do not add global `raw/` or
`processed/` layers.

If an external dataset has its own subsets, keep the source files and all data
processing results inside the relevant subset:

```text
data/
  openskyDataset/
    subset_1/
      set_1.csv
      set_1.parquet
      set_1_aircraft.csv
      set_1_sensors.csv
      NOTE.txt
      LICENSE.txt
      civil_data_for_realtime_web_show/
        20250510.csv
        report/
    subset_2/
    subset_3/
```

Important interpretation:

- `openskyDataset` represents one dataset.
- `subset_1` is internal structure of `openskyDataset`, not a project-level dataset split.
- Original source files live directly inside the corresponding subset.
- Storage-format conversions such as `set_1.parquet` are treated as source
  inputs for downstream work and stay beside the original CSV.
- Every processing result belongs in a clearly named child directory of the
  dataset or subset that produced it.
- Project-level code should identify the dataset as `openskyDataset`.
- Dataset-internal subsets should be handled by the dataset reader/config.

### Derived Data

Canonical caches, preprocessing artifacts, and reusable training samples stay
under their source dataset or subset. Use descriptive result-directory names
and manifests to identify the processing stage and configuration.

Example:

```text
data/
  openskyDataset/
    subset_1/
      canonical_v1_<config_hash>/
        tracks.parquet
        manifest.yaml
      kalman_singer_dt5_win24_<config_hash>/
        tracks.parquet
        manifest.yaml
        input60_pred10_xyz_zscore_<config_hash>/
          train.npz
          val.npz
          test.npz
          manifest.yaml
```

## Outputs Directory

Experiment outputs should not be mixed into `data/`.

Use:

```text
outputs/
  runs/
  checkpoints/
  predictions/
  reports/
  exported_figures/
```

Prediction results, metrics, reports, and exported figures are experiment outputs,
not source data assets.

## Source Code Directory

Keep the standard Python src layout:

```text
src/
  trajectory_prediction/
    schemas/
    datasets/
    preprocess/
      methods/
    samples/
    models/
      lstm/
      patchtst/
      kalman/
      imm_kalman/
    training/
    inference/
    evaluation/
    artifacts/
    experiments/
    cli/
```

Keep `src/trajectory_prediction/` even though it repeats the project name.
This is the preferred Python package layout and supports stable imports:

```python
from trajectory_prediction.datasets import ...
```

## Apps Directory

Web visualization should be separate from the prediction core.

Use:

```text
apps/
  experiment_viewer/
    routes/
    services/
    templates/
    static/
      assets/
      css/
      js/
```

The web viewer is an offline experiment result browser.
It should read artifacts such as:

- processed tracks
- predictions
- metrics
- manifests

It should not own training, preprocessing, or inference logic.

## Small Data / Development Cases

Do not require fixed `tiny/` or `sample/` directories.

For development, the user may select any dataset or internal subset explicitly.
Derived directory names or their `manifest.yaml` must state the processing role.
Do not infer data stage only from a vague folder name like `test_data`.

## Naming Preferences

Prefer:

- lower snake_case for new project-controlled names
- readable variant name plus config hash
- `manifest.yaml` for artifact metadata

Examples:

```text
kalman_singer_dt5_win24_ab12cd
input60_pred10_xyz_zscore_ef34aa
lstm_hidden128_layers2_lr1e3_45cc90
```

Existing raw dataset names may keep their current names, for example
`openskyDataset`, because they may reflect original data naming.

## Migration Rule

Do not copy old projects wholesale into this project.

Old code sources:

- `D:/code/LSTM`
- `D:/code/patchTST`
- `D:/code/web_trajectory_visualizer/kalman`
- `D:/code/web_trajectory_visualizer/IMMkalman`

Treat them as references. Migrate only cleaned, useful core logic into the new
`src/trajectory_prediction/` structure.

## Practical Rule For Future Codex Work

When unsure where something belongs, decide by role:

- data asset -> `data/`
- experiment output -> `outputs/`
- reusable prediction platform code -> `src/trajectory_prediction/`
- web result browser code -> `apps/experiment_viewer/`
- user-facing explanation -> `docs/`
- Codex/agent working notes -> `skills/`
