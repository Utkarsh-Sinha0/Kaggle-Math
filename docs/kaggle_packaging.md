# Kaggle Packaging Flow

## Goal

Package code, prompts, configs, and docs into a dataset-style folder that can be attached to a Kaggle notebook, while keeping model weights in a separate dataset mount.

## Build the bundle

```bash
python scripts/package_kaggle_bundle.py --zip
```

Outputs:

- `output/kaggle_bundle/`
- `output/kaggle_bundle.zip` when `--zip` is supplied

## What gets rewritten

- `configs/model.yaml`
  - `local_model_path` becomes the Kaggle model mount
  - `local_files_only` becomes `true`
- `configs/runtime.yaml`
  - `backend` becomes `transformers`

## Expected Kaggle mounts

- Code bundle dataset: `/kaggle/input/aimo3-bundle`
- Model dataset: `/kaggle/input/aimo3-model`

## Included notebook

The bundle contains `notebooks/kaggle_submission_baseline.ipynb`, which:

- adds the bundle root to `sys.path`
- sets `AIMO3_KAGGLE=1`
- points `AIMO3_MODEL_PATH` at the model dataset mount
- loads the copied configs from the bundle
- runs `solve_one` on a smoke prompt

Replace the smoke prompt with the real competition input path once the final notebook I/O is fixed.
