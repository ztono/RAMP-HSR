# Availability statement

GitHub and Zenodo publication links are pending.

**Data availability.** The repository manifests (including repository URLs,
file paths, source hashes, and verified commit SHAs where recovered), sample
metadata, training/test split identities, and aggregate
route-audit outputs are publicly available at [GitHub URL] (archived version:
https://doi.org/[Zenodo DOI]). Because the source files are drawn from third-party
GitHub repositories under their original licenses, we do not redistribute the
raw source code.

**Code availability.** The implementation of RAMP-HSR, risk-route extraction
scripts, baseline configurations (MINT/STAR), and all evaluation scripts are
publicly available at [GitHub URL] (archived version:
https://doi.org/[Zenodo DOI]).

## Mapping

| Declaration item | Release location |
| --- | --- |
| Repository URLs and commit SHAs | `data/repositories.jsonl.gz` |
| File paths and available source hashes | `data/files.jsonl.gz` |
| Sample metadata | `data/pools/` |
| Training/test split identities | `data/repositories.jsonl.gz`, `data/splits/` |
| Aggregate route-audit outputs | `data/aggregates/route_audit.json` |
| RAMP-HSR implementation | `ramp_hsr/`, `scripts/train.py` |
| Risk-route extraction | `scripts/extract_risk_routes.py`, `configs/routes/` |
| MINT/STAR configurations | `configs/baselines/` |
| Evaluation scripts | `scripts/`, `EVALUATION.md` |

