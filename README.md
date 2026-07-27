# Payment Fraud Research Sandbox

This repository is being rebuilt as a reproducible sandbox for generating fake
payment transactions and, in a later phase, evaluating fraud-detection models.
The legacy reports, model, notebooks, and generated CSVs were removed because
they could not be reproduced reliably and the previous evaluation leaked target
and future information.

Read the [repository audit](AUDIT.md) for the findings that motivated the reset
and the [synthetic-data research notes](SYNTHETIC_DATA_RESEARCH.md) for the design
options and proposed transaction schema.

## Current scope

The first Phase 1 generator is now implemented for synthetic card-authorization
events. It creates persistent customers, accounts, merchants, explicit fraud
campaigns, and chronologically ordered transactions from one seed. Model training
remains intentionally out of scope until the simulator has been calibrated and its
sample-size policy has been agreed.

```text
.
├── generator.py                  # deterministic event simulator and CLI
├── tests/                        # determinism and data-integrity tests
├── data/.gitkeep                 # generated data belongs here, but is not tracked
├── AUDIT.md                      # evidence and restart roadmap
├── SYNTHETIC_DATA_RESEARCH.md    # source review and proposed sample design
├── pyproject.toml                # package metadata and dependency policy
└── requirements.txt              # convenience development installation
```

## Set up a development environment

Python 3.10 through 3.14 is supported by the project metadata.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

Runtime dependencies are deliberately limited to NumPy and pandas. ML,
visualization, and notebook dependencies will be added as optional groups only
when the corresponding phase has a tested implementation.

## Generate a sample

```bash
python generator.py \
  --seed 42 \
  --start-date 2025-01-01 \
  --days 90 \
  --customers 500 \
  --merchants 300 \
  --fraud-rate 0.002 \
  --output-dir data
```

This writes five untracked CSV files under `data/`: customers, accounts, merchants,
campaigns, and transactions. Fraud counts are allocated from the requested rate
and scenario weights using deterministic largest-remainder rounding. The simulator
fails before saving if identifier, foreign-key, chronological, amount, provenance,
or fraud-count validation fails.

## Next milestone

Calibrate and harden the simulator before starting ML work:

1. agree on fraud prevalence, scenario weights, and minimum independent campaigns
   per chronological split;
2. compare the generated distributions and dependencies with verified references
   or a lawful aggregate seed dataset;
3. expand validation for campaign overlap, authorization state, and temporal drift;
4. only then build the causal feature pipeline and baseline models.

The detailed proposal is in [SYNTHETIC_DATA_RESEARCH.md](SYNTHETIC_DATA_RESEARCH.md).
