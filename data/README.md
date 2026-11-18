# Data artifacts

No third-party source files, source-bearing prompts, generated completions,
token sequences that reconstruct source, or complete source-content maps are
included. Repository and sample identities, source hashes, occurrence offsets,
training/test split identities, and aggregate route-audit metrics are retained.

## Metadata and splits

`pools/` contains compact metadata for every occurrence in the main source pools:

| Pool | Train | Test |
| --- | ---: | ---: |
| Python | 2,230,488 | 437,434 |
| TypeScript | 6,564,536 | 1,447,823 |

Each row stores only an integer `file_id`, source character positions, identifier
and suffix lengths, and line/column. The file dictionary is `files.jsonl.gz`.
Original source-shard order is preserved, so language, split, shard index, and
zero-based row position identify the original cache row. The index records the
schema and exact counts. These are the common raw source-example pools for Qwen
and StarCoder2 reported in paper Table B9. Model-specific tokenization partitions
are computed from these same source examples.

`splits/` also retains training repository identities and exact file SHA-256 sets
for isolation checks. Full-pool row totals and training repository/file-hash sets
are checked against the retained archive contracts. `splits/index.json` specifies
the exported scope. Source-bearing originals remain in the experiment archives.

`benchmarks/` separately records the external execution benchmark's repository
identities, verified revisions, and paths/hashes of evaluated files. Revision
provenance distinguishes archived Git HEADs from recovered content matches.
`benchmarks/execrepobench_bundle.json` provides the fixed dataset revision,
download URLs, and SHA-256 values for the official `repos.zip` and sample file.
Use that bundle for execution and keep downloaded source local.

## Repository provenance

`repositories.jsonl.gz` contains the original roster of 3,600 Python and 3,600
TypeScript repositories, their URLs, and available verified commit pins. `files.jsonl.gz` contains relative paths and normalized UTF-8 source hashes.
Each recovered pin reproduces every referenced file in that repository. Missing
pins remain null. Source reconstruction checks file and prefix hashes using the
original sampler's UTF-8 decoding and newline normalization.

Clone each required upstream repository locally under `local/repos/<repo_id>`
and run:

```bash
python -m scripts.reconstruct_samples \
  --repositories data/repositories.jsonl.gz \
  --samples data/pools/python/test.shard_0.jsonl.gz \
  --files data/files.jsonl.gz \
  --clones local/repos --output local/python_samples.jsonl
```

Repeat for each required language, split, and shard. The command expands compact
rows through the shared file dictionary before reconstructing source spans.

Reconstruction checks the original full-file and prefix hashes. It uses the
sampler's UTF-8 replacement decoding and universal newline normalization, and
fails if an upstream snapshot differs. Source-bearing outputs remain local.

## Route-audit aggregates

`aggregates/route_audit.json` records the four Risk route rows from paper Table 2
on the full test pools. Values retain the paper's reporting precision; these are
reported aggregates, not a new experimental run. Test sample counts follow
Table B9. Baseline results and per-sample predictions are excluded.

Third-party repositories retain their original licenses. No new license is
inferred for upstream source or model weights by this metadata export.
