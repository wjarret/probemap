# SPEC.md — `probemap`

**Audience:** the coding agent building this package. This document is the
full design brief. Build to it; where it says "verify at build time," do
that verification before writing dependent code, and flag back to the user
if a named resource doesn't pan out as described.

**Course context:** graduate statistical computing course, Track B (Python
package). MVP demo deadline: **2026-11-23**. That is roughly 4 weeks from
spec date (2026-10-26). This is a hard external constraint — see
[Timeline risk](#timeline-risk-and-fallback) at the end before planning
your own work schedule.

---

## 1. What this package is (and is not)

`probemap` is a Python package for training machine-learning classifiers
on per-nucleotide RNA chemical-probing data (SHAPE-MaP, RING-MaP, RNP-MaP
reactivity and mutation-rate profiles).

**It is:** a modular ML framework. Users build per-nucleotide feature
tables from one or more independent feature modules, train classifiers on
any subset of those modules, and optionally combine separately-trained
per-module models via a weighted-average ensemble.

**It is not:** a reimplementation of deltaSHAPE, dStruct, diffBUM-HMM, or
DiffScan. Those tools' *statistics* (Z-factor tests, HMMs, scan
statistics) are explicitly out of scope. The *idea* behind one of them —
comparing two reactivity profiles to find changed nucleotides — is instead
distilled into one feature module (`diffreact`, §4.4) that computes a
simplified difference/Z-score signal and feeds it into the ML pipeline as
a feature, not a standalone statistical test. Do not build a general
differential-testing framework; build one feature column.

This scope decision was deliberate (confirmed with the user over the
originally broader framing) — do not re-expand it.

## 2. Package identity

- **Name:** `probemap`
- **Import:** `import probemap`
- **Repo:** `https://github.com/wjarret/probemap` (public)
- **Install:** `pip install git+https://github.com/wjarret/probemap`
- **Python:** 3.10+
- **License:** MIT
- **OS support:** Windows, macOS, Linux — verify every dependency choice
  installs cleanly on all three before locking it into `pyproject.toml`.
- **Packaging:** `pyproject.toml` with a standard build backend
  (hatchling or setuptools — your choice, just keep it boring).

## 3. Two prediction tasks (both in scope, fully)

The user explicitly chose to build **both** tasks fully, accepting that
this is more work than a typical 4-week MVP normally carries. Scope your
time accordingly and read the [timeline risk note](#timeline-risk-and-fallback)
before committing to a schedule.

### Task A: paired vs. unpaired (secondary structure)

Binary label per nucleotide: is it base-paired in the accepted/reference
secondary structure, or not.

### Task B: binding-site vs. non-site (protein footprinting)

Binary label per nucleotide: is it inside a region called as
protein-bound (RNP-MaP functional hub / crosslink-enriched region) in the
source study, or not.

Both tasks must be reachable through the **same API** (`featurize` →
`train` → [`ensemble`] → `evaluate`), selected via a `task` argument, not
via separate code paths. Default feature-module sets differ per task
(§4), but the module registry and model-training code must be task-agnostic.

## 4. Feature modules

`featurize(profiles, modules=[...])` builds a per-nucleotide feature
table. Each module is an independent, swappable unit. A user can request
any subset.

Implement exactly these four:

### 4.1 `reactivity` — normalized reactivity

Per-nucleotide normalized SHAPE/DMS reactivity. Default normalization:
boxplot ("2–8%") normalization as used in the Siegfried/Smola SHAPE-MaP
pipeline (Smola et al. 2015, *Nat Protoc*, doi:10.1038/nprot.2015.103) —
exclude top/bottom percentiles as outliers, normalize to the mean of the
next band. Make the normalization method swappable (`method="boxplot"` by
default; leave room for `"winsor_zscore"` as an alternative) but ship only
boxplot normalization for the MVP.

### 4.2 `seqcontext` — local sequence context

One-hot encoded nucleotide identity in a window around each position
(default window: ±3 nt, i.e. 7 columns × 4 bases = 28 features). Window
size configurable.

### 4.3 `ringcorr` — RING-MaP correlation features

Derived from RingMapper pairwise correlation output (see §5.1 for parser).
Per-nucleotide summary statistics:
- correlation degree (number of significant partners)
- max correlation strength (e.g. max Z-score or MI across partners)
- mean partner genomic distance

This is the **default third module for Task A**. It is optional for Task
B (not all binding-site datasets ship RING-MaP data) — `featurize` must
not error if `ringcorr` is requested but the input lacks RING-MaP data; it
should raise a clear, specific exception telling the user which input was
missing, not a generic KeyError.

### 4.4 `diffreact` — differential reactivity score

A simplified deltaSHAPE-style per-nucleotide difference score between two
conditions (e.g. ±protein, ±ligand, in vivo/in vitro). Compute a smoothed
difference (sliding-window mean difference, default window 3 nt) and a
standard-score normalization of that difference — i.e. the *statistic*
deltaSHAPE uses, not deltaSHAPE's significance-calling/thresholding logic.
This is the **default module for Task B** (binding-site detection is
fundamentally a ±protein comparison). Not applicable to Task A unless the
user explicitly supplies a second condition.

**Default module sets** (overridable by the user):
- Task A: `["reactivity", "seqcontext", "ringcorr"]`
- Task B: `["reactivity", "seqcontext", "diffreact"]`

## 5. Data ingestion

`probemap.io` reads raw files into a standard per-nucleotide table
(columns at minimum: `rna_id`, `position`, `nucleotide`, `reactivity`,
`std_err`, plus modality-specific columns). Support these input formats:

### 5.1 Parsers to implement

1. **ShapeMapper2 `_profile.txt`** — the primary real-world format.
   Columns include `Nucleotide, Sequence, Reactivity_profile, Std_err,
   HQ_profile, ...`. Validate your parser against real ShapeMapper2
   example output (the repo ships example data at
   `github.com/Weeks-UNC/shapemapper2` under its `example_data/` — pull a
   small fixture from there for parsing tests, verify the exact path at
   build time since it may move between versions).
2. **RingMapper correlation output** — pairwise correlation file feeding
   `ringcorr` (§4.3). Check RingMapper's own example output
   (`github.com/Weeks-UNC/ringmapper`) for the exact column schema and
   build a fixture from it the same way.
3. **RNP-MaP reactivity profiles** — RNP-MaP data is produced through a
   ShapeMapper2-derived pipeline with crosslink-adduct rate columns,
   generally compatible with the `_profile.txt` schema plus condition
   metadata (±crosslinker). Confirm the exact column differences against
   the Weidmann et al. 2020 *Nat Biotechnol* (doi:
   10.1038/s41587-020-0709-7) supplementary data or associated GEO
   accession (candidate: GSE152483 — verify at build time) before writing
   a separate parser; if the schema is profile.txt-compatible, reuse the
   ShapeMapper2 parser rather than duplicating it.
4. **Generic CSV/TSV** — a documented minimal schema
   (`rna_id,position,nucleotide,reactivity,mutation_rate,std_err`
   required columns, others optional) so simulated test data and
   non-ShapeMapper2 users have a straightforward path in. Document this
   schema clearly in the README since it's the lowest-friction entry
   point for new users.

### 5.2 Reagent/batch metadata and QC diagnostic (optional, non-blocking)

When ingesting profiles, allow the user to optionally record which
chemical probing reagent was used (e.g. 1M7, NAI, DMS) per sample as batch
metadata. Implement `probemap.qc.check_batch_effects(table, by="reagent")`
as an **optional diagnostic function**, not a required pipeline step — it
checks whether reactivity distributions differ systematically across
reagent/batch groups (e.g. Kolmogorov–Smirnov test or simple
distributional summary per group) and returns/plots a warning-style report
a user can consult before training, to catch a confound before it
contaminates a model. This must not be wired into `featurize`, `train`,
or any required code path — it's a standalone QC helper the user calls if
they want to.

## 6. Training, ensembling, evaluation

### 6.1 `train(features, labels, task, modules, backend, ...)`

- `task`: `"paired"` or `"binding_site"`.
- `modules`: list of feature-module names to use for this model. A single
  module → "independent" training on that module alone. Multiple modules
  → "mixed" training on their concatenated feature columns in one model.
  Both must go through the same `train()` call; "mixed" is not a separate
  function, just a longer `modules` list.
- `backend`: **required, no default** — either `"sklearn"` or
  `"autogluon"`. Do not silently pick one; raise if omitted. Keep the two
  backends behind a small common interface (`fit`, `predict_proba`) so
  `ensemble`/`evaluate` don't need to know which backend trained a given
  model.
  - sklearn backend: use `HistGradientBoostingClassifier` or
    `RandomForestClassifier` as the default estimator (laptop-friendly, no
    GPU, light dependency footprint — appropriate for the course's
    laptop-first requirement).
  - autogluon backend: heavier dependency, appropriate for the HPC path;
    keep it an optional extra (`pip install probemap[autogluon]`) so the
    base install stays light on Windows/macOS/Linux CI.
- `n_jobs` (or equivalent resource-control parameter): expose parallelism
  control on `train`/`evaluate` so cross-validation folds can be
  parallelized on an HPC allocation (§8).

### 6.2 `ensemble(models, method="weighted_average")`

Combine predicted probabilities from multiple separately-trained
per-module models (i.e., multiple outputs of `train()`, each with a
different single-module or mixed-module `modules` list). MVP implements
**weighted averaging only**:
- default: uniform weights
- optional: weights fit on a held-out validation split (e.g. a simple
  constrained least-squares or logistic regression over out-of-fold
  per-model probabilities) to let the ensemble favor stronger modules

Do not implement stacking for the MVP — explicitly out of scope per the
user's decision, to keep ensemble logic simple and testable.

"Independent, mixed, or ensemble" is therefore satisfied as: independent
= one `train()` call with one module; mixed = one `train()` call with
several modules concatenated; ensemble = several `train()` calls (one per
module or module-set) combined via `ensemble()`.

### 6.3 `evaluate(model_or_ensemble, data)`

- Cross-validation **split by RNA, not by nucleotide** — nucleotides from
  the same RNA must never span both train and test folds within a single
  CV split. Implement as a grouped K-fold (group = `rna_id`).
- Metrics: AUROC and AUPRC.
- Must also support evaluating an `ensemble()` output through the same
  call signature as a single `train()` output.

### 6.4 Baseline comparison

Implement a trivial reactivity-threshold classifier (predict
paired/binding-site purely from a reactivity cutoff, no learning) and run
it through the same `evaluate()` path. This is a required correctness
check (§9), not optional — every reported model result should be
comparable against this baseline.

## 7. Datasets

Public data only, plus a small simulated dataset for unit tests.

### 7.1 Task A (paired vs. unpaired)

**Primary: RNA Mapping Database (RMDB).** Now mirrored as a static site
backed by a public GitHub repo (the original `rmdb.stanford.edu` Django
site was retired; RDAT files are distributed via GitHub releases —
`eterna`, `puzzle`, `riboswitches`, `rna-structures`, `general`). This
gives curated reactivity profiles with known/accepted secondary
structures for many RNAs (tRNAs, 5S rRNA, riboswitches, group I/II
introns) in one consistent RDAT format — ideal for a labeled benchmark.
**Verify at build time:** locate the current GitHub repo/org for RMDB,
confirm RDAT format parsing requirements, and pick 3–5 structurally
diverse, well-validated entries (e.g. a riboswitch, a tRNA, a larger
ribozyme) for the MVP rather than trying to ingest the whole archive.

**Secondary / fallback: Siegfried et al. 2014** (*Nat Methods*,
doi:10.1038/nmeth.3029, PMC4259394) SHAPE-MaP of the HIV-1 genome and E.
coli rRNA, validated against >90% of accepted base pairs in RNAs of known
structure. Data availability section cites SRA accession **SRX554885** —
verify at build time exactly which runs/RNAs this covers (HIV-1 vs.
rRNA), since this was not independently confirmed during spec-writing.
Known secondary structures for 16S/23S rRNA are available from the
Comparative RNA Web (CRW) site — cross-check accessibility before relying
on it.

### 7.2 Task B (binding-site vs. non-site)

**Primary: RNP-MaP original dataset.** Weidmann, Mustoe, Jariwala,
Calabrese, Weeks, "Analysis of RNA–protein networks with RNP-MaP defines
functional hubs on RNA," *Nat Biotechnol* 39, 347–356 (2021; online 2020),
doi:10.1038/s41587-020-0709-7. Candidate GEO accession **GSE152483**
("RNP-MaP: In-cell analysis of protein interaction networks defines
functional hubs in RNA," human/mouse) — **verify this accession at build
time** before relying on it; it was identified via search, not directly
read from the paper. The study's functional-hub calls on the XIST lncRNA
(known protein-binding repeat regions A–F) are the natural source of
positive labels; non-hub regions are negatives.

If GSE152483 doesn't check out as described, fall back to the paper's own
supplementary data tables (binding-site coordinates are typically
reported directly in supplementary materials for MaP-based footprinting
papers) rather than guessing at a different accession.

### 7.3 Simulated test data

A small synthetic dataset for unit tests: generate reactivity profiles
with known ground-truth pairing/binding states (e.g. simulate reactivity
as low-mean for "structured"/paired positions and high-mean for
unpaired/bound positions, plus noise) so featurize/train/evaluate can be
tested without network access or real data dependencies. Ship this under
`tests/data/` as static fixtures, generated once by a documented script,
not regenerated at test time (keeps tests deterministic).

## 8. HPC support

No special code path beyond the `n_jobs`/resource-control parameter in
§6.1. The package is pure Python/scikit-learn (plus optional AutoGluon
extra) and should run identically on a cluster node. Deliverable:

- `n_jobs` parameter on `train()` and `evaluate()` controlling
  cross-validation fold parallelism (thread or process pool — your
  choice, document which).
- A documented example SLURM batch script in `docs/hpc_example.md` or
  similar, showing the same `train()`/`evaluate()` calls submitted as a
  cluster job, plus a note on running the same calls interactively in a
  Jupyter session on an HPC allocation.

This is intentionally light — don't build cluster-scheduling logic into
the package itself.

## 9. Correctness checks (required, not stretch)

1. **Unit tests on simulated data** (§7.3) covering `featurize`, `train`
   (both backends, both tasks), `ensemble`, `evaluate`, and the QC
   diagnostic.
2. **Parsing tests against real ShapeMapper2 and RingMapper example
   output** — pull small fixtures from the upstream repos (§5.1) and
   assert the parser produces the expected standard table shape/values.
3. **Baseline comparison** — every example/tutorial that reports a
   trained-model AUROC/AUPRC must also report the reactivity-threshold
   baseline (§6.4) on the same split, so a reader can see the model beats
   a trivial rule.

Use `pytest`. All tests must pass on Windows, macOS, and Linux in CI.

## 10. CI

GitHub Actions workflow: matrix over `windows-latest`, `macos-latest`,
`ubuntu-latest` × Python 3.10, 3.11, 3.12 (or whatever the current stable
set is at build time). Run `pytest` on every push/PR. Keep the AutoGluon
extra out of the default CI matrix dependency install (test it in a
separate, slower job or mark its tests to skip when the extra isn't
installed) so base CI stays fast across all three OSes.

## 11. Stretch goals (explicitly NOT MVP)

Do not build these unless the MVP (§3–10) is fully done with time to
spare:
- Lightweight interactive UI (Streamlit or Shiny)
- Additional probing modalities beyond SHAPE-MaP/RING-MaP/RNP-MaP
- Pretrained models shipped with the package

## Timeline risk and fallback

The user explicitly chose to build **both prediction tasks fully** (§3),
acknowledging this is more work than a typical 4-week course MVP. If,
partway through the build, it's clear both tasks can't be finished with
real data, full test coverage, and baseline comparisons by 2026-11-23:

- **Fallback:** keep Task A (paired/unpaired) as the fully-built,
  fully-tested primary task. Reduce Task B (binding-site) to a thin
  demonstration — the same API, one small dataset, lighter test coverage
  — to prove the module/task abstraction generalizes without finishing a
  second full validation pass.

Flag this tradeoff to the user explicitly if you have to invoke it; don't
silently drop Task B coverage without saying so.
