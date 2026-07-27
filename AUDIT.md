# Repository audit and restart plan

Audit date: 2026-07-26

> Historical note: the counts and file inventory below describe the repository at
> audit time. The subsequent cleanup removed the non-reproducible CSVs, notebooks,
> generated reports, visualization script, and serialized model from version
> control. The underlying generator findings remain the basis for Phase 1.

## Executive summary

The repository is a useful proof of concept, but its current dataset and evaluation
cannot support claims about production fraud detection. The checked-in sample is
small (3,463 transactions), has a 7.85% fraud rate rather than the documented 0.2%
target, and is dominated by one synthetic scenario. More importantly, feature
engineering uses labels and future observations before the train/test split. This
leaks the answer into the model and makes the reported 99%+ ROC-AUC unreliable.

The recommended restart order is:

1. make dataset generation deterministic, configurable, internally consistent,
   and testable;
2. define a time-aware, leakage-free feature contract and evaluation protocol;
3. establish simple baselines and business metrics before comparing advanced
   models.

## What exists today

| Area | Current state |
| --- | --- |
| Synthetic data | Customer, terminal, and transaction generation in `generator.py`; four intended fraud scenarios |
| Checked-in sample | 100 customers, 200 terminals, 3,463 transactions covering 2024-01-01 through 2024-01-30 |
| Analysis | Exploratory plots, feature engineering, five classifier families, threshold analysis, and a real-time simulation in one script |
| Artifacts | CSV data, notebooks, PNG/HTML reports, and a pickled XGBoost model are committed |
| Tests | Three generator tests and one feature-engineering test; the suite currently fails during collection |
| Packaging/automation | Flat scripts and an unbounded `requirements.txt`; no CI, lock file, package metadata, or command-line configuration |

## Dataset findings

The following profile was calculated directly from the committed CSV files:

| Check | Result |
| --- | --- |
| Transactions | 3,463 |
| Customers / terminals | 100 / 200 |
| Fraud labels | 272 (7.85%) |
| Scenario distribution | legitimate 3,191; scenario 1: 2; scenario 2: 0; scenario 3: 33; scenario 4: 237 |
| Missing cells / duplicate full rows | 0 / 0 |
| Duplicate transaction IDs | present; IDs are not unique after injected cash-out transactions |
| Timestamp consistency | 237 rows disagree between `TX_DATETIME` and `TX_TIME_SECONDS` |

### Critical generation issues

1. **The configured target is not enforced.** `target_fraud_rate` and
   `scenario_distribution` are descriptive configuration only. The generator's
   outcome is driven by separate probabilities and by appending three to five
   cash-out rows for selected customers on every day. The sample therefore has
   roughly 39 times the advertised fraud prevalence and 87% of fraud belongs to
   scenario 4.
2. **Injected cash-out records are internally inconsistent.** They copy an old
   transaction, update `TX_TIME_SECONDS`, but do not recompute `TX_DATETIME`.
   They also inherit `TRANSACTION_ID`. All 237 scenario-4 records in the sample
   have timestamp disagreements, and transaction IDs are not unique.
3. **Scenario 2 silently produces no examples.** It is absent from the committed
   sample. Its window uses `day + 45`, despite iterating daily, while applying a
   very small per-transaction probability. Tests do not require every configured
   scenario to be represented.
4. **Scenario semantics are weak.** Scenario 1 labels rare naturally generated
   high amounts rather than injecting an attack, while scenario 3 only relabels
   existing online nighttime activity. Scenario 4 creates transactions at
   arbitrary terminals without modeling authorization, balance, geography, or a
   compromise event. This encourages a classifier to learn generator shortcuts.
5. **Reproducibility is partial.** Several helpers reset global random state, but
   fraud injection consumes global NumPy state without accepting a seed or RNG.
   Importing the generator also initializes four `pandarallel` workers even though
   the main generation path uses ordinary `DataFrame.apply` and Python loops.
6. **The scale is insufficient for rare-event evaluation.** At a genuine 0.2%
   prevalence, 3,463 rows would contain about seven positives. A larger time span,
   more entities, and many independent seeds are needed for stable per-scenario
   measurements.

## ML and evaluation findings

### Blockers

1. **Direct target leakage:** `terminal_risk_score` is calculated from
   `TX_FRAUD` across the complete dataset before splitting. Every test record,
   including its label, contributes to its terminal's feature.
2. **Future leakage:** customer-terminal counts and same-day transaction counts
   use all records, including later activity. A real-time decision would only have
   history strictly before the transaction.
3. **Random rather than temporal validation:** stratified random splitting mixes
   the same customers, terminals, compromise windows, and adjacent transactions
   between train and test. Synthetic burst transactions can therefore straddle
   both sets. A chronological holdout with a gap is the minimum realistic design;
   entity or campaign holdouts should also be reported.
4. **Feature preprocessing can discard valid columns:** the transformer selects
   only `int64` and `float64`. Boolean and other numeric dtypes (including one-hot
   columns on current pandas versions) may be silently dropped.
5. **Model selection and threshold selection reuse the test set.** The best model
   is selected by test ROC-AUC, and the threshold is then optimized on the same
   labels. Train, validation, and final untouched test periods must be distinct.

### Other reliability issues

- ROC-AUC is emphasized despite severe class imbalance. Report precision-recall
  AUC, recall at a review-capacity/false-positive budget, precision at k, expected
  cost, calibration, and per-scenario confidence intervals.
- SMOTE is applied indiscriminately to engineered indicator/count features and is
  combined with class-weighting in several models. This can create implausible
  feature vectors and obscures which imbalance treatment helps.
- No trivial baselines are recorded. Amount rules, terminal-risk rules, and a
  leakage-free logistic regression should precede boosted trees.
- The pickled model lacks schema, generator version, training window, dependency
  versions, metrics, threshold, and provenance. Pickle is also unsafe to load from
  untrusted sources.
- The real-time simulation ignores its model argument and hard-codes
  `models/xgboost_model.pkl`. It passes NumPy rows to a pipeline fitted with named
  DataFrame columns and reads a nonexistent `is_flagged` result key. A broad
  exception catches these failures and allows the overall script to appear to
  complete.

## Engineering and test findings

- `pytest -q` fails at collection because the tests import a nonexistent
  `generate_transactions` function; the implementation exports
  `generate_transactions_table` and `generate_dataset` instead.
- The feature test uses independently sampled fraud labels and scenarios, so the
  two target columns can contradict each other. Random fixtures are not seeded.
- No tests cover deterministic output, requested prevalence tolerance, scenario
  balance, referential integrity, unique IDs, timestamp derivation, chronological
  ordering, or absence of leakage.
- Generated reports, notebooks, data, and model binaries are mixed with source.
  It is unclear which artifacts are canonical or how each can be reproduced.
- Dependencies have only lower bounds. This makes the environment drift over time
  and is especially risky for serialized scikit-learn/XGBoost pipelines.
- The README clone URL is a placeholder and its performance claims are not tied to
  a reproducible run or an evaluation artifact.

## Recommended staged roadmap

### Phase 1 — trustworthy synthetic data (next)

1. Introduce a typed configuration and CLI for entity counts, date range, fraud
   prevalence, scenario mix, and seed; pass a local `numpy.random.Generator`
   through every generation function.
2. Separate legitimate event generation from fraud campaign injection. Model
   compromise events with start/end times and create rows whose observable fields
   are internally consistent.
3. Assign transaction IDs only after all events are created; derive datetime/day
   fields from one canonical timestamp; validate foreign keys, ranges, ordering,
   uniqueness, and label/scenario consistency before saving.
4. Treat prevalence and scenario weights as generation constraints with documented
   tolerance. Fail loudly if a requested scenario cannot be generated.
5. Produce small deterministic fixtures for tests and larger data as reproducible
   build artifacts rather than manually curated CSVs.

**Phase 1 acceptance criteria:** the same seed produces byte-equivalent tables;
different seeds produce different samples; all integrity checks pass; observed
fraud rate and scenario mix are within stated tolerance; each scenario has enough
examples for the intended split; and the full test suite passes.

### Phase 2 — leakage-free ML baseline

1. Define an as-of feature API: each feature may use only information available
   before the authorization timestamp. Fit aggregate encoders on training history
   and update them causally for validation/test events.
2. Use chronological train/validation/test periods, with optional campaign and
   unseen-entity stress tests. Keep the final test labels untouched until model and
   threshold choices are frozen.
3. Add rule-based, dummy, and regularized logistic baselines. Compare class
   weighting and resampling as separate experiments before tree ensembles.
4. Version datasets, configuration, feature schema, metrics, threshold, and model
   metadata together. Add fixed-seed reproducibility checks.

### Phase 3 — realistic experimentation

1. Increase population and duration, run multiple seeds, and include concept
   drift, delayed labels, recurring merchants/customers, declined transactions,
   and campaign-level correlations.
2. Evaluate calibration and operational trade-offs, including review capacity,
   false declines, fraud loss, and per-scenario/segment performance with confidence
   intervals.
3. Only then compare gradient boosting, anomaly detection, sequence models, or
   graph approaches against the established baselines.

## Proposed immediate work item

Start with Phase 1 as a focused change: refactor generation around a seeded config,
fix identifier/timestamp integrity, enforce scenario counts, and replace the broken
tests with deterministic invariants. Do not tune the current models before this is
complete; doing so would optimize against leaked, inconsistent labels.
