# English maintenance scope

Completed on 2026-09-11 at the user's request. Active development and evaluation
now target English retrieval. This cleanup removes 238 net Python lines relative
to the pre-cleanup workspace, including the retired pilot generator.

## Changes

- Removed the language-specific sentence separator from recursive Markdown
  chunking. ASCII sentence punctuation, Markdown boundaries and lossless fallback
  splitting remain. Algorithm identity is now `markdown-v2`.
- Removed the T2 public importer, default download entries, preparation,
  queue/finalization branches and language-specific tokenization audit.
  Active public tracks are SciFact, Bright-Pro Stack Overflow/Robotics and MuSiQue.
- Removed the old pilot data generator. The generated dataset, labels and review
  packets remain unchanged; a historical generator is available in the snapshot.
- Removed six language-specific test parameters and replaced other test questions,
  titles and bodies with English or general Unicode fixtures. Added the previous
  Markdown algorithm to the existing index-upgrade regression.
- Removed language-count diagnostics and retired the language-specific tokenizer
  roadmap item. Newly generated review instructions use English.

UTF-8 input, normalization, combining characters, emoji, byte-versus-character
coordinates and source integrity remain supported and tested. No language filter
rejects retained datasets. Existing model, retrieval and dependency choices are
unchanged by this cleanup.

## Index compatibility

Existing snapshots remain readable. On the next normal `arkb index`, the new
chunking fingerprint prevents silently reusing an older recursive snapshot as the
current configuration. A new snapshot is published; compatible embedding inputs
can be reused. The regression verifies both publication and embedding reuse.
The fixture's English title changes its expected revision/chunk hashes; the
identity implementation and persisted schema did not change.

## Retained data and history

T2's 118,605 documents, fixed 500 queries, qrels, raw downloads, multipart archive
and historical measurements remain intact. Existing v1/pilot/core-intake data and
review packets remain available. Their earlier measurements do not become new
English benchmark results after this cleanup.

The [pre-cleanup snapshot](../evaluation/experiments/artifacts/pre-english-scope-20260911.tar.gz)
and its [manifest](../evaluation/experiments/artifacts/pre-english-scope-20260911.tar.manifest.json)
retain 274 files, including the original source, tests, removed generator and P4
delivery metadata. It is a historical artifact outside active module discovery.
All 124 entries in the original P4 delivery index remain recoverable from this
snapshot plus the unchanged dataset archives. Historical indices describe those
frozen bytes, not subsequently edited files in the checkout. Restore the snapshot
only into a separate directory when reproducing the old delivery.

## Verification

- Deterministic suite: **1,058 passed**, 59 integration cases excluded.
- Actual cached tokenizer and local Ollama token/embedding/context checks:
  **24 passed**. Other model/Agent quality runs were not repeated.
- Current-code SciFact regression: **300/300** complete-corpus BM25 rankings and
  metrics unchanged, maximum metric difference 0.
- Active raw-input audit: **27 locked files**, 178,291 corpus units, 516 selected
  retrieval queries and 200 original MuSiQue variants verified.
- Retained T2 dataset hashes and the complete original P4 artifact inventory
  verified; active Python source/tests/scripts contain no retired language
  branches or fixtures. All 191 active Python modules parse successfully.

See the [cleanup verification record](../evaluation/audits/20260911-english-scope-cleanup.json)
for changed-file hashes and retained evidence. Earlier P0–P4 work in the checkout
was preserved; this cleanup has not been committed or pushed.
