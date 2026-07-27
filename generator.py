"""Deterministic event simulator for synthetic card-payment authorizations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

SCENARIO_WEIGHTS = {
    "cnp_account_compromise": 0.35,
    "stolen_card": 0.25,
    "cash_out_burst": 0.20,
    "terminal_compromise": 0.20,
}
CHANNELS = ("card_present", "ecommerce", "atm", "wallet")
MERCHANT_CATEGORIES = ("grocery", "fuel", "restaurant", "retail", "travel", "cash")


@dataclass(frozen=True)
class SimulationConfig:
    """Inputs that fully determine a simulation run."""

    seed: int = 42
    start_date: str = "2025-01-01"
    days: int = 90
    n_customers: int = 500
    n_merchants: int = 300
    target_fraud_rate: float = 0.002
    scenario_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        weights = self.scenario_weights or SCENARIO_WEIGHTS
        if self.days < 1 or self.n_customers < 1 or self.n_merchants < 4:
            raise ValueError("days/customers must be positive and at least four merchants are required")
        if not 0 < self.target_fraud_rate < 1:
            raise ValueError("target_fraud_rate must be between zero and one")
        if set(weights) != set(SCENARIO_WEIGHTS):
            raise ValueError(f"scenario_weights must contain {sorted(SCENARIO_WEIGHTS)}")
        if any(weight < 0 for weight in weights.values()) or not np.isclose(sum(weights.values()), 1):
            raise ValueError("scenario_weights must be nonnegative and sum to one")
        try:
            timestamp = pd.Timestamp(self.start_date)
        except (TypeError, ValueError) as error:
            raise ValueError("start_date must be a valid timestamp") from error
        if pd.isna(timestamp):
            raise ValueError("start_date must be a valid timestamp")

    @property
    def weights(self) -> Mapping[str, float]:
        return self.scenario_weights or SCENARIO_WEIGHTS


@dataclass(frozen=True)
class SimulationResult:
    customers: pd.DataFrame
    accounts: pd.DataFrame
    merchants: pd.DataFrame
    campaigns: pd.DataFrame
    transactions: pd.DataFrame


class PaymentSimulator:
    """Generate legitimate activity first, then explicit fraud campaigns."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        start = pd.Timestamp(config.start_date)
        self.start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        self.end = self.start + pd.Timedelta(days=config.days)

    def run(self) -> SimulationResult:
        customers = self._customers()
        accounts = self._accounts(customers)
        merchants = self._merchants()
        legitimate = self._legitimate_transactions(customers, accounts, merchants)
        campaigns, fraudulent = self._fraud_transactions(accounts, merchants, len(legitimate))
        transactions = pd.concat([legitimate, fraudulent], ignore_index=True)
        transactions = transactions.sort_values("event_timestamp", kind="stable").reset_index(drop=True)
        transactions.insert(0, "transaction_id", [f"tx_{value:09d}" for value in range(len(transactions))])
        result = SimulationResult(customers, accounts, merchants, campaigns, transactions)
        validate_result(result, self.config)
        return result

    def _customers(self) -> pd.DataFrame:
        count = self.config.n_customers
        segments = self.rng.choice(["everyday", "affluent", "student"], count, p=[0.65, 0.20, 0.15])
        rates = np.where(segments == "affluent", 2.4, np.where(segments == "student", 0.7, 1.3))
        typical = np.where(segments == "affluent", 140, np.where(segments == "student", 28, 65))
        return pd.DataFrame({
            "customer_id": [f"cust_{i:06d}" for i in range(count)],
            "segment": segments,
            "home_region": self.rng.choice(["north", "south", "east", "west", "central"], count),
            "mean_daily_transactions": rates,
            "typical_amount": typical.astype(float),
        })

    def _accounts(self, customers: pd.DataFrame) -> pd.DataFrame:
        count = len(customers)
        return pd.DataFrame({
            "account_id": [f"acct_{i:06d}" for i in range(count)],
            "customer_id": customers["customer_id"],
            "credit_limit": np.round(customers["typical_amount"].to_numpy() * self.rng.uniform(15, 35, count), 2),
            "status": "active",
        })

    def _merchants(self) -> pd.DataFrame:
        count = self.config.n_merchants
        category = self.rng.choice(MERCHANT_CATEGORIES, count, p=[.22, .12, .18, .25, .08, .15])
        channel = np.empty(count, dtype=object)
        for i, value in enumerate(category):
            if value == "cash":
                channel[i] = "atm"
            else:
                channel[i] = self.rng.choice(["card_present", "ecommerce", "wallet"], p=[.62, .28, .10])
        # All fraud scenarios must remain feasible even in small deterministic
        # fixtures; reserve one acceptance point for every supported channel.
        channel[:4] = CHANNELS
        category[:4] = ["retail", "retail", "cash", "retail"]
        return pd.DataFrame({
            "merchant_id": [f"merch_{i:06d}" for i in range(count)],
            "terminal_id": [f"term_{i:06d}" for i in range(count)],
            "merchant_category": category,
            "channel": channel,
            "merchant_region": self.rng.choice(["north", "south", "east", "west", "central"], count),
        })

    def _legitimate_transactions(self, customers, accounts, merchants) -> pd.DataFrame:
        rows: list[dict] = []
        account_ids = accounts["account_id"].to_numpy()
        limits = accounts["credit_limit"].to_numpy()
        for index, customer in customers.iterrows():
            count = self.rng.poisson(customer.mean_daily_transactions * self.config.days)
            seconds = np.sort(self.rng.integers(0, self.config.days * 86400, count))
            merchant_indices = self.rng.integers(0, len(merchants), count)
            amounts = np.maximum(0.5, self.rng.lognormal(np.log(customer.typical_amount), 0.65, count))
            for second, merchant_index, amount in zip(seconds, merchant_indices, amounts):
                merchant = merchants.iloc[merchant_index]
                amount = round(float(amount), 2)
                rows.append(self._transaction_row(
                    self.start + pd.Timedelta(seconds=int(second)), customer.customer_id,
                    account_ids[index], merchant, amount, limits[index], False, None, None
                ))
        return pd.DataFrame(rows)

    def _fraud_transactions(self, accounts, merchants, legitimate_count):
        fraud_count = round(
            legitimate_count
            * self.config.target_fraud_rate
            / (1 - self.config.target_fraud_rate)
        )
        counts = allocate_counts(fraud_count, self.config.weights)
        campaigns: list[dict] = []
        rows: list[dict] = []
        for scenario, count in counts.items():
            if count == 0:
                continue
            campaign_count = min(count, max(1, int(np.ceil(count / 5))))
            allocations = allocate_evenly(count, campaign_count)
            for number, allocation in enumerate(allocations):
                campaign_id = f"camp_{scenario}_{number:04d}"
                account = accounts.iloc[int(self.rng.integers(0, len(accounts)))]
                if scenario == "terminal_compromise":
                    merchant = merchants.iloc[int(self.rng.integers(0, len(merchants)))]
                elif scenario == "cash_out_burst":
                    choices = merchants.index[merchants.channel == "atm"]
                    merchant = merchants.loc[int(self.rng.choice(choices))]
                elif scenario == "cnp_account_compromise":
                    choices = merchants.index[merchants.channel == "ecommerce"]
                    merchant = merchants.loc[int(self.rng.choice(choices))]
                else:
                    choices = merchants.index[merchants.channel == "card_present"]
                    merchant = merchants.loc[int(self.rng.choice(choices))]
                start_second = int(self.rng.integers(0, max(1, self.config.days * 86400 - 3600)))
                start = self.start + pd.Timedelta(seconds=start_second)
                duration_minutes = 60 if scenario == "cash_out_burst" else int(self.rng.integers(120, 2880))
                campaigns.append({
                    "campaign_id": campaign_id,
                    "fraud_scenario": scenario,
                    "start_timestamp": start,
                    "end_timestamp": min(start + pd.Timedelta(minutes=duration_minutes), self.end),
                })
                remaining_seconds = int((self.end - start).total_seconds())
                offset_horizon = min(duration_minutes * 60, remaining_seconds)
                for offset in np.sort(self.rng.integers(0, offset_horizon, allocation)):
                    timestamp = start + pd.Timedelta(seconds=int(offset))
                    if scenario == "terminal_compromise":
                        victim = accounts.iloc[int(self.rng.integers(0, len(accounts)))]
                    else:
                        victim = account
                    typical = float(victim.credit_limit) / 25
                    multiplier = 2.5 if scenario == "cash_out_burst" else 1.5
                    amount = round(float(self.rng.lognormal(np.log(typical * multiplier), 0.55)), 2)
                    rows.append(self._transaction_row(
                        timestamp, victim.customer_id, victim.account_id, merchant, amount,
                        victim.credit_limit, True, scenario, campaign_id
                    ))
        campaign_columns = (
            "campaign_id", "fraud_scenario", "start_timestamp", "end_timestamp"
        )
        return pd.DataFrame(campaigns, columns=campaign_columns), pd.DataFrame(rows)

    def _transaction_row(self, timestamp, customer_id, account_id, merchant, amount, limit,
                         is_fraud, scenario, campaign_id):
        decline_probability = 0.02 + (0.55 if amount > limit else 0)
        result = "declined" if self.rng.random() < decline_probability else "approved"
        entry_mode = {"card_present": "chip", "ecommerce": "remote", "atm": "chip", "wallet": "token"}[merchant.channel]
        authentication = {"card_present": "pin", "ecommerce": "three_ds", "atm": "pin", "wallet": "biometric"}[merchant.channel]
        return {
            "event_timestamp": timestamp,
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant.merchant_id,
            "terminal_id": merchant.terminal_id,
            "channel": merchant.channel,
            "merchant_category": merchant.merchant_category,
            "amount": amount,
            "currency": "USD",
            "merchant_region": merchant.merchant_region,
            "entry_mode": entry_mode,
            "authentication_type": authentication,
            "authorization_result": result,
            "is_fraud": bool(is_fraud),
            "fraud_scenario": scenario,
            "campaign_id": campaign_id,
        }


def allocate_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate an exact total with deterministic largest-remainder rounding."""
    raw = {name: total * weight for name, weight in weights.items()}
    result = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:remaining]:
        result[name] += 1
    return result


def allocate_evenly(total: int, groups: int) -> list[int]:
    return [total // groups + (index < total % groups) for index in range(groups)]


def validate_result(result: SimulationResult, config: SimulationConfig) -> None:
    """Fail before saving if identifiers, labels, or configured rates are invalid."""
    tx = result.transactions
    entity_keys = {
        "customers": (result.customers, "customer_id"),
        "accounts": (result.accounts, "account_id"),
        "merchants": (result.merchants, "merchant_id"),
        "campaigns": (result.campaigns, "campaign_id"),
    }
    for table_name, (frame, key) in entity_keys.items():
        if frame[key].isna().any() or not frame[key].is_unique:
            raise ValueError(f"{table_name} must have non-null unique {key} values")
    if tx.empty or not tx.transaction_id.is_unique:
        raise ValueError("transactions must be nonempty with unique IDs")
    if not tx.event_timestamp.is_monotonic_increasing:
        raise ValueError("transactions must be chronologically ordered")
    start = PaymentSimulator(config).start
    end = start + pd.Timedelta(days=config.days)
    if (tx.event_timestamp < start).any() or (tx.event_timestamp >= end).any():
        raise ValueError("transaction timestamps must be within the simulation horizon")
    if (tx.amount <= 0).any() or not np.allclose(tx.amount * 100, np.round(tx.amount * 100)):
        raise ValueError("amounts must be positive and currency-rounded")
    if not set(tx.customer_id).issubset(set(result.customers.customer_id)):
        raise ValueError("transaction customer foreign key is invalid")
    if not set(tx.account_id).issubset(set(result.accounts.account_id)):
        raise ValueError("transaction account foreign key is invalid")
    if not set(tx.merchant_id).issubset(set(result.merchants.merchant_id)):
        raise ValueError("transaction merchant foreign key is invalid")
    if not set(tx.terminal_id).issubset(set(result.merchants.terminal_id)):
        raise ValueError("transaction terminal foreign key is invalid")
    account_owners = result.accounts.set_index("account_id").customer_id
    if not (tx.customer_id.to_numpy() == tx.account_id.map(account_owners).to_numpy()).all():
        raise ValueError("transaction customer does not own its account")
    merchant_terminals = result.merchants.set_index("merchant_id").terminal_id
    if not (tx.terminal_id.to_numpy() == tx.merchant_id.map(merchant_terminals).to_numpy()).all():
        raise ValueError("transaction terminal does not belong to its merchant")
    unlabeled = tx.fraud_scenario.isna() & tx.campaign_id.isna()
    if not (unlabeled == ~tx.is_fraud).all():
        raise ValueError("fraud labels and provenance are inconsistent")
    expected = round((~tx.is_fraud).sum() * config.target_fraud_rate / (1 - config.target_fraud_rate))
    if int(tx.is_fraud.sum()) != expected:
        raise ValueError("fraud prevalence does not match configured allocation")
    campaign_ids = set(result.campaigns.campaign_id)
    if not set(tx.loc[tx.is_fraud, "campaign_id"]).issubset(campaign_ids):
        raise ValueError("fraud campaign foreign key is invalid")
    campaigns = result.campaigns.set_index("campaign_id")
    if not campaigns.empty:
        invalid_window = (
            (campaigns.start_timestamp < start)
            | (campaigns.start_timestamp >= campaigns.end_timestamp)
            | (campaigns.end_timestamp > end)
        )
        if invalid_window.any():
            raise ValueError("campaign timestamps must form valid simulation windows")
        fraud = tx.loc[tx.is_fraud]
        joined = fraud.join(
            campaigns[["fraud_scenario", "start_timestamp", "end_timestamp"]],
            on="campaign_id",
            rsuffix="_campaign",
        )
        if not (joined.fraud_scenario == joined.fraud_scenario_campaign).all():
            raise ValueError("transaction fraud scenario does not match its campaign")
        if not (
            (joined.event_timestamp >= joined.start_timestamp)
            & (joined.event_timestamp <= joined.end_timestamp)
        ).all():
            raise ValueError("fraud transaction falls outside its campaign window")


def save_result(result: SimulationResult, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("customers", "accounts", "merchants", "campaigns", "transactions"):
        frame = getattr(result, name)
        frame.to_csv(directory / f"{name}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--merchants", type=int, default=300)
    parser.add_argument("--fraud-rate", type=float, default=0.002)
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        seed=args.seed, start_date=args.start_date, days=args.days,
        n_customers=args.customers, n_merchants=args.merchants,
        target_fraud_rate=args.fraud_rate,
    )
    result = PaymentSimulator(config).run()
    save_result(result, args.output_dir)
    print(f"Generated {len(result.transactions):,} transactions from {asdict(config)}")


if __name__ == "__main__":
    main()
