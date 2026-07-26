# Research notes: building a better synthetic transaction sample

Research date: 2026-07-26

## Scope and research constraint

This review focuses on data design, not model selection. Online access was
attempted during the review, but both the browsing service and direct HTTPS access
were blocked by the environment (HTTP 401 and proxy 403 responses respectively).
The links below are therefore a curated reading queue of primary project pages and
papers, based on their published designs; their latest versions and licenses must
be verified when network access is available. No external dataset has been copied
into this repository.

## Reference approaches to evaluate

### 1. Fraud Detection Handbook simulator

- [Simulated dataset chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html)
- [Handbook source repository](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)

This is the closest reference to the repository's current customer/terminal model.
Its useful pattern is to build ordinary customer behavior first, associate
customers with nearby terminals, and apply several explicit fraud mechanisms over
time. We should use it as a transparent baseline, while fixing limitations that
matter here: richer merchant/payment context, explicit compromise campaigns,
configurable prevalence, and strict as-of semantics.

### 2. PaySim

- [PaySim paper DOI](https://doi.org/10.1109/EMSS.2016.7887909)
- [PaySim project repository](https://github.com/EdgarLopezPhD/PaySim)

PaySim is an agent-based mobile-money simulator derived from aggregate behavior.
It is useful for studying balances, transfer types, and fraud embedded in a
transaction process rather than merely relabeling independent rows. Its domain is
mobile money, however, so its transaction types and fraud mechanisms should not be
presented as card-payment behavior without adaptation.

### 3. BankSim

- [BankSim paper](https://arxiv.org/abs/1409.6241)

BankSim is a bank-payment simulation reference for generating normal and fraudulent
behavior from agents. The main lesson for this project is methodological: generate
transactions from persistent actors and evolving state, then represent fraud as
behavioral episodes. A row-independent random sampler will not reproduce velocity,
novelty, recurrence, or coordinated campaigns.

### 4. Real-data benchmarks for calibration, not redistribution

- [ULB credit-card fraud benchmark paper](https://doi.org/10.1109/CIBCB.2015.7350894)
- [IEEE-CIS fraud detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection)

Real benchmarks can help sanity-check prevalence, class overlap, temporal drift,
and attainable metrics. They should not be merged blindly with synthetic rows, and
their licenses and access terms must be reviewed before downloading or committing
anything. The anonymized ULB features also cannot directly define a realistic
business-facing schema.

### 5. General synthetic-data frameworks

- [Synthetic Data Vault documentation](https://docs.sdv.dev/sdv)
- [Synthetic Data Vault source](https://github.com/sdv-dev/SDV)

Statistical synthesizers may become useful after a lawful seed dataset exists.
They do not, by themselves, provide fraud ground truth or guarantee that rare
attack campaigns are preserved. Any such framework must be evaluated for temporal
and relational fidelity, privacy leakage, rare-class coverage, and reproducibility
before adoption.

## Recommended approach for this repository

Use a configurable, event-based simulator rather than training a tabular generator
on the flawed legacy CSV. The simulator should have four layers.

### Layer A — persistent entities

- customers: segment, home region, account age, normal activity schedule, spending
  distribution, and risk-independent behavior parameters;
- payment instruments/accounts: owner, activation/expiry, limit or balance, status,
  and compromise state;
- merchants/terminals: merchant category, channel, region, currency, opening hours,
  and stable identifiers;
- fraud actors/campaigns: objective, compromised instruments, active interval,
  targeted channels/merchants, and attack parameters.

Fraud propensity must not be encoded directly into a customer field that will later
be handed to a model. Hidden simulator state may drive events, while model-visible
features must be observations available at authorization time.

### Layer B — legitimate authorization events

Generate activity chronologically from customer-specific processes. Amount,
channel, merchant category, geography, and time should be conditionally related
rather than sampled independently. Include both approved and declined attempts so
the model's eventual decision target is not confused with the fraud label.

### Layer C — explicit fraud campaigns

Start with a small, auditable taxonomy:

1. card-not-present account compromise;
2. stolen-card or impossible-travel card-present use;
3. cash-out burst with velocity and balance/limit effects;
4. merchant or terminal compromise affecting several customers.

Each fraud row should reference a campaign ID that is retained for evaluation but
excluded from model inputs. Campaigns should modify the event stream through the
same authorization rules as legitimate attempts. Counts should be allocated from
the configured prevalence and scenario weights, with a clear rounding policy and
an error when constraints cannot be met.

### Layer D — deterministic validation

Use one local `numpy.random.Generator` derived from a declared seed. After all
events are created, assign unique transaction IDs and derive every calendar field
from one canonical UTC timestamp. Validate:

- primary-key uniqueness and foreign-key coverage;
- chronological ordering and exact timestamp-derived fields;
- nonnegative, currency-rounded amounts and valid categorical domains;
- `is_fraud == 0` iff the fraud scenario/campaign fields are empty;
- configured prevalence and scenario mix within documented tolerance;
- a minimum number of campaigns and positive rows in each eventual data split;
- byte-equivalent normalized output for repeated runs with the same seed.

## Proposed first schema

| Column | Purpose | Available to a model? |
| --- | --- | --- |
| `transaction_id` | Unique event identifier | no |
| `event_timestamp` | Canonical UTC authorization time | yes |
| `customer_id` | Persistent customer reference | aggregation key only |
| `account_id` | Payment account/instrument reference | aggregation key only |
| `merchant_id`, `terminal_id` | Acceptance entity references | aggregation keys only |
| `channel` | card-present, e-commerce, ATM, wallet | yes |
| `merchant_category` | Coarse purchase category | yes |
| `amount`, `currency` | Requested transaction value | yes |
| `customer_region`, `merchant_region` | Coarse locations, not raw personal coordinates | yes |
| `entry_mode`, `authentication_type` | Observable authorization context | yes |
| `authorization_result` | approved or declined outcome | only for later events |
| `is_fraud` | Ground-truth target, potentially delayed | target only |
| `fraud_scenario`, `campaign_id` | Simulator provenance and sliced evaluation | no |

Derived velocity, novelty, and risk aggregates do not belong in the raw table.
They should be computed later by a causal feature builder using only events strictly
earlier than the scored authorization.

## Sample sizing and split strategy

Do not choose a row count in isolation. Work backward from the rarest scenario and
the final chronological test window. A candidate configuration is acceptable only
if every evaluation slice contains enough independent campaigns—not merely many
rows from one burst—to estimate uncertainty.

For fast development, keep a tiny fixed fixture with dozens of entities and enough
forced examples to test every invariant. For experiments, generate longer runs and
multiple seeds. Split by time into train, validation, and untouched test periods;
add a gap for label delay, and report campaign-disjoint and unseen-entity stress
tests. Never rebalance the validation or test prevalence.

## Decision gates before implementation

The initial implementation assumes **card authorization** events and uses simulator
ground truth (`is_fraud`) as the eventual classification target. It includes both
approved and declined attempts, but does not treat the authorization decision as a
fraud label. These assumptions should be confirmed before calibration work.

1. Confirm the intended payment domain: card authorization, bank transfer, mobile
   money, or a deliberately documented mixture.
2. Choose the operational target: confirmed fraud, chargeback, suspicious attempt,
   or authorization decision. These are not interchangeable.
3. Define acceptable prevalence/scenario tolerances and minimum campaigns per
   split.
4. Verify the reference links, source versions, and licenses when network access is
   restored.
5. Implement the smallest event simulator and its validators before adding any ML
   dependency or notebook.
