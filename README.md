# RAMP-HSR

Research artifacts accompanying the paper:

> **Risk Routes for Explainable Automated Software Engineering: Diagnosing and Repairing Identifier-Token Mismatches in Code Completion**  
> Junzheng Chen and Zhaoyuan Zhang  
> Published in **[Automated Software Engineering (ASE)](https://link.springer.com/journal/10515)**, Springer.

## Overview

**RAMP-HSR** stands for **Route-Audited Hidden-State Repair**. It studies
file-local identifier completion at supplied identifier-start or
identifier-internal cursors, using a frozen code language model.

The completion workflow separates two components:

- **Risk-route diagnosis:** a fixed readout over selected hidden-state dimensions
  across model layers measures identifier-token mismatch risk.
- **Hidden-state repair:** a sample-local residual MLP with a learned strength
  gate edits the final post-normalization hidden state. Predictions are decoded
  through the frozen language-model head.

Route scores are independent audit readouts and do not enter the editor. Training
combines target recovery with preservation of base-correct predictions.

The paper evaluates Qwen2.5-Coder-1.5B and StarCoder2-3B on Python and TypeScript.
It also investigates observed-identifier monitoring. This release focuses on the
completion-mode implementation and its route-audit workflow.

## Repository contents

| Location | Contents |
| --- | --- |
| `ramp_hsr/` | Editor, training components, model interface, route readouts, and metrics |
| `scripts/train.py` | Main editor training entry point |
| `scripts/extract_risk_routes.py` | Training-side risk-route discovery |
| `scripts/build_*.py` | Input preparation and hidden-state cache construction |
| `scripts/evaluate_*.py`, `scripts/summarize_*.py` | Evaluation and result aggregation |
| `configs/ramp_hsr.json` | Main architecture and experiment settings |
| `configs/routes/` | Frozen model- and language-specific route policies |
| `configs/baselines/` | MINT/STAR configurations and pinned upstream references |
| `data/` | Repository manifests, sample metadata, splits, and aggregate route-audit outputs |

MINT and STAR are provided as **configuration files only**. Their implementations
are obtained from the upstream repositories referenced in those files.

## Installation

Use Python 3.10 or later with a suitable PyTorch installation:

```bash
python -m pip install -e .
```

For optional CodeBLEU evaluation:

```bash
python -m pip install -e '.[completion]'
```

Base-model weights and tokenizers must be downloaded separately. Store local
models, reconstructed source, caches, and generated outputs under `local/`.

## Getting started

### 1. Prepare local inputs

The published metadata identifies repository revisions, source files, and
identifier positions. Reconstruct source-bearing inputs from local upstream
clones using `scripts.reconstruct_samples`, then prepare hidden-state caches with
`scripts.build_hidden_cache`.

```bash
python -m scripts.reconstruct_samples --help
python -m scripts.build_hidden_cache extract-shard --help
python -m scripts.build_hidden_cache finalize --help
```

See [data/README.md](data/README.md) for metadata formats and reconstruction
instructions. Reconstruction verifies source hashes against the manifests.

### 2. Train the editor

With training caches and a local base model prepared:

```bash
python -m scripts.train \
  --packed-dataset local/hidden/train \
  --base-model local/Qwen2.5-Coder-1.5B \
  --output local/editor.pt
```

The main settings are recorded in [configs/ramp_hsr.json](configs/ramp_hsr.json).
The released editor uses width 256 for both the residual and strength branches.

### 3. Extract or load risk routes

Use training-side records for route discovery:

```bash
python -m scripts.extract_risk_routes --help
```

Frozen route policies are provided in `configs/routes/`. For evaluation commands
that use runtime route readouts, select a policy through
`SRLCC_NEURON_RISK_CONFIG`, as described in [EVALUATION.md](EVALUATION.md).

### 4. Evaluate

For next-token repair evaluation with a prepared test cache:

```bash
python -m scripts.evaluate_tokens \
  --packed-dataset local/hidden/test \
  --base-model local/Qwen2.5-Coder-1.5B \
  --checkpoint local/editor.pt \
  --output local/token_metrics.json
```

The evaluation scripts cover token recovery and damage, complete-identifier
generation, context and cursor controls, route detection and calibration,
writeback and rollout dynamics, residual transport and projection, and
repository-native execution. See [EVALUATION.md](EVALUATION.md) for entry points
and protocol definitions.

## Data and code availability

The repository contains:

- Repository URLs, available verified commit SHAs, file paths, and source hashes.
- Sample metadata for the Python and TypeScript pools.
- Training/test split identities.
- Reported route-audit aggregates in `data/aggregates/route_audit.json`.
- The released RAMP-HSR components, route-extraction scripts, MINT/STAR
  configurations, and evaluation scripts.

The repository roster contains **3,000 training and 600 evaluation repositories
per language**. Raw third-party source files remain with their upstream
repositories under their original licenses and are not redistributed here.
Commit SHAs are supplied where content-verified; unresolved entries remain null.

See [AVAILABILITY.md](AVAILABILITY.md) for the availability statements and
GitHub/Zenodo publication links.
