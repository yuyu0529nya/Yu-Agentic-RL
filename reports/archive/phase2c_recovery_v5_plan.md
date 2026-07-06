# Phase 2C Recovery-Prefix v5 Plan

## Goal

Recovery-Prefix v5 targets the three stable failure families exposed by the v4 wider eval:

- `all_reservations_lookup_from_user_id`: task 18 refused to inspect reservations after the user provided a valid user id.
- `booking_payment_plan_single_passenger`: task 25 lost passenger count/payment grounding and never called `book_reservation`.
- `duration_requires_search_direct`: task 44 used `get_flight_status` when it needed schedule/duration evidence from `search_direct_flight`.

The intended test is whether targeted recovery SFT can move these failures without regressing task 2 and task 16.

## Dataset

- Train: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_train.jsonl`
- Valid: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_valid.jsonl`
- Heldout behavior probes: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_heldout.jsonl`
- Corrections: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_corrections.jsonl`
- Manifest: `data/recovery_prefix/tau2_airline_recovery_prefix_manifest_v5_2048.json`
- Dataset report: `reports/recovery_prefix_dataset_v5_2048.md`

Current local validation:

- Train rows: `2155`
- Raw correction rows: `126`
- Oversample factor: `16`
- Bad loss masks: `0`
- Duplicate IDs: `0`
- Max messages: `41`

## Training

Default script:

```bash
bash scripts/run_recovery_prefix_sft_4090_autodl.sh
```

Default configuration:

- Model: `Qwen2.5-7B-Instruct`
- Method: 4-bit QLoRA SFT
- Version: `v5`
- Max sequence length: `2048`
- Steps: `650`
- LoRA rank: `16`
- LR: `2e-4`
- Auto shutdown: off by default

Expected 4090 memory:

- QLoRA SFT training: about `16-21 GB`
- vLLM/tau2 evaluation afterward: about `22-23 GB`

## Evaluation Order

1. Behavior probe on heldout next-action samples.
2. Targeted tau2 eval on `18,25,44`.
3. Regression tau2 eval on `2,16`.
4. Only if targeted/regression results improve, run a wider sweep.

## Success Criteria

Minimum useful result:

- task 18 or task 44 improves from failure to pass, while task 2 and task 16 do not regress.

Strong result:

- at least two of `18,25,44` pass, and regression tasks `2,16` remain pass.

If v5 trains well but tau2 pass does not improve, the next algorithm step should be constrained tool decoding or PRM rerank rather than blindly adding more SFT epochs.
