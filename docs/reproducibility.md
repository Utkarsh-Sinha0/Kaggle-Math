# Reproducibility Guide

## Principles

- Keep prompts, configs, and code in Git
- Record commit hash and config hash for every run
- Separate lightweight runtime artifacts from raw training data
- Re-run promising configurations on the same eval slice before promoting them

## Minimum run record

- Git commit hash
- Config hash
- Data slice id
- Runtime in seconds
- Pass@1, majority@N, selector@N
- Invalid-answer rate
- Timeout/crash rate
- Short notes on what changed

## Promotion rule

Only promote a change to the Kaggle notebook path after the AWS baseline is stable and the change survives at least one repeated run on the same internal slice.

