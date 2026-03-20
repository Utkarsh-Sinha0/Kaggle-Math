# AIMO3 Autoresearch Program

You are running bounded autonomous research for the AIMO3 Nemotron Nano solver.

## Goal

Improve `selector_at_n` on the small public eval slice without increasing invalid answers or blowing the GPU-hour budget.

## Only editable file

- `configs/research_policy.yaml`

Do not edit Python source, packaging, compliance docs, or deployment scripts.

## Metric

Primary score:

`selector_at_n - invalid_answer_rate - timeout_rate`

## Guardrails

- Stop if the budget ledger shows less than 5 GPU-hours of headroom.
- Do not increase `sample_count` beyond 8.
- Do not disable the final one-integer guarantee.
- Do not introduce fine-tuning or dataset mutation.

## Loop

1. Read `configs/research_policy.yaml`
2. Pick one safe knob change
3. Run the small eval slice
4. Keep only improvements that are repeatable and budget-safe
