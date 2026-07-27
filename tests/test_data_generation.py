"""Tests for deterministic synthetic payment generation."""

import pandas as pd
import pytest

from generator import (
    PaymentSimulator, SimulationConfig, allocate_counts, save_result, validate_result,
)


def config(**changes):
    values = dict(seed=7, start_date="2025-01-01", days=10, n_customers=20,
                  n_merchants=30, target_fraud_rate=0.08)
    values.update(changes)
    return SimulationConfig(**values)


def test_same_seed_produces_identical_tables():
    first = PaymentSimulator(config()).run()
    second = PaymentSimulator(config()).run()
    for table in ("customers", "accounts", "merchants", "campaigns", "transactions"):
        pd.testing.assert_frame_equal(getattr(first, table), getattr(second, table))


def test_different_seed_changes_transactions():
    first = PaymentSimulator(config(seed=1)).run().transactions
    second = PaymentSimulator(config(seed=2)).run().transactions
    assert not first.equals(second)


def test_transaction_integrity_and_scenario_allocation():
    result = PaymentSimulator(config()).run()
    tx = result.transactions
    assert tx.transaction_id.is_unique
    assert tx.event_timestamp.is_monotonic_increasing
    assert set(tx.loc[tx.is_fraud, "fraud_scenario"]) == {
        "cnp_account_compromise", "stolen_card", "cash_out_burst", "terminal_compromise"
    }
    assert tx.loc[~tx.is_fraud, ["fraud_scenario", "campaign_id"]].isna().all().all()
    legitimate = int((~tx.is_fraud).sum())
    expected = round(legitimate * 0.08 / 0.92)
    assert int(tx.is_fraud.sum()) == expected


def test_small_sample_does_not_force_a_fraud_transaction():
    result = PaymentSimulator(config(
        seed=0, days=1, n_customers=1, n_merchants=4,
        target_fraud_rate=0.002,
    )).run()

    assert len(result.transactions) == 1
    assert not result.transactions.is_fraud.any()
    assert result.campaigns.empty


def test_campaign_offsets_stay_within_remaining_simulation_horizon():
    simulation_config = config(days=1, target_fraud_rate=0.8)
    result = PaymentSimulator(simulation_config).run()
    fraud = result.transactions[result.transactions.is_fraud]
    campaign_ends = result.campaigns.set_index("campaign_id").end_timestamp

    assert (fraud.event_timestamp < PaymentSimulator(simulation_config).end).all()
    assert (fraud.event_timestamp != PaymentSimulator(simulation_config).end - pd.Timedelta(microseconds=1)).all()
    assert all(
        row.event_timestamp <= campaign_ends[row.campaign_id]
        for row in fraud.itertuples()
    )


def test_save_result_writes_all_tables(tmp_path):
    result = PaymentSimulator(config()).run()
    save_result(result, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "customers.csv", "accounts.csv", "merchants.csv", "campaigns.csv", "transactions.csv"
    }


def test_saved_output_is_byte_reproducible(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    save_result(PaymentSimulator(config()).run(), first_dir)
    save_result(PaymentSimulator(config()).run(), second_dir)
    for filename in ("customers.csv", "accounts.csv", "merchants.csv", "campaigns.csv", "transactions.csv"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="fraud_rate"):
        config(target_fraud_rate=0)
    with pytest.raises(ValueError, match="sum to one"):
        config(scenario_weights={name: 0.5 for name in allocate_counts(1, {
            "cnp_account_compromise": .25, "stolen_card": .25,
            "cash_out_burst": .25, "terminal_compromise": .25,
        })})
    with pytest.raises(ValueError, match="start_date"):
        config(start_date="not-a-date")


def test_timezone_aware_start_date_is_normalized_to_utc():
    simulation = PaymentSimulator(config(start_date="2025-01-01T02:00:00+02:00"))

    assert simulation.start == pd.Timestamp("2025-01-01T00:00:00Z")


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("customer_id", "cust_missing", "customer foreign key"),
        ("terminal_id", "term_missing", "terminal foreign key"),
    ],
)
def test_validation_rejects_broken_transaction_foreign_keys(column, replacement, message):
    simulation_config = config()
    result = PaymentSimulator(simulation_config).run()
    result.transactions.loc[0, column] = replacement

    with pytest.raises(ValueError, match=message):
        validate_result(result, simulation_config)


def test_validation_rejects_campaign_scenario_mismatch():
    simulation_config = config()
    result = PaymentSimulator(simulation_config).run()
    fraud_index = result.transactions.index[result.transactions.is_fraud][0]
    current = result.transactions.loc[fraud_index, "fraud_scenario"]
    replacement = next(name for name in allocate_counts(0, simulation_config.weights) if name != current)
    result.transactions.loc[fraud_index, "fraud_scenario"] = replacement

    with pytest.raises(ValueError, match="does not match"):
        validate_result(result, simulation_config)
