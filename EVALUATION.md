# Evaluation entry points

All commands run from this repository root after installation. Each script
supports `python -m scripts.<name> --help`; provide explicit local input/output
paths. Source-bearing inputs and generated predictions belong under `local/`.

| Evaluation or preparation | Entry point |
| --- | --- |
| Recover local samples from verified commits | `reconstruct_samples` |
| Cache frozen post-normalization hidden states | `build_hidden_cache` |
| Stage the frozen norm/head for cached-data runs | `stage_lm_head_cache` |
| Train-only risk-route discovery | `extract_risk_routes` |
| Next-token exactness, error recovery, damage, Net | `evaluate_tokens` |
| Frozen route detection and confidence curves | `evaluate_route_detection` |
| Calibration-only route/confidence threshold selection | `evaluate_operating_points` |
| Same-cursor complete-identifier generation | `evaluate_identifiers`, `summarize_identifiers` |
| Cursor-boundary control | `build_cursor_controls`, `summarize_cursor_controls` |
| Context-length control | `build_context_controls`, `summarize_context_controls` |
| Original/aligned public-prefix protocols | `build_public_inputs`, `evaluate_public_protocol`, `summarize_public_protocol` |
| One first-token intervention, frozen-base continuation | `evaluate_continuation` |
| Post-writeback route changes | `evaluate_route_shift` |
| Two-identifier reset/reinjection route dynamics | `evaluate_downstream_routes` |
| Residual transport and route-region projection audits | `evaluate_residual_transport`, `evaluate_route_projection` |
| Repository-native execution and compilation | `evaluate_execution` |
| Paired execution metrics and repository bootstrap | `summarize_execution` |
| Repository and exact-file isolation | `audit_isolation`, `exclude_duplicate_files` |

Route policies are selected explicitly through `SRLCC_NEURON_RISK_CONFIG` when a
runtime route readout is needed. The policy files under `configs/routes/` preserve
their original node indices and aggregation rules. They never feed the MLP.

Token exactness measures recovery of the single repository reference token.
Complete-identifier exactness uses the unchanged source cursor and requires an
observed right boundary; no-identifier and censored generations remain in the
denominator. Cursor shifts, longer contexts, and original/aligned public prompts
are reported as separately named conditions.

Continuation applies one edit at token zero and then continues with the frozen
base model. Route movement is an audit outcome. It is not an inference gate or
evidence of autonomous semantic correction. Execution follows the recorded
reference-scaffold/reference-length protocol and repository-native tests;
its Pass@1 is not an original-prefix leaderboard score.

Uncertainty estimates use paired bootstrap replicates at repository level.
Select calibration examples from training-side records and compute their model
scores locally. Thresholds are selected using training-side records
only and applied unchanged to the test cohort. Evaluation scripts record their
bootstrap settings and protocol IDs in local outputs.
