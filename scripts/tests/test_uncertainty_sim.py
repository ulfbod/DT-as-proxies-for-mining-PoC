"""Tests for uncertainty_sim.py — Task A and Task C TDD."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import uncertainty_sim as us

POC_SCENARIO_KEY = "scenario_2_idt"


# ── Task A tests ──────────────────────────────────────────────────────────────

def test_poc_scenario_exists():
    assert POC_SCENARIO_KEY in us.UNCERTAINTY_SCENARIOS


def test_poc_provider_labels():
    cfg = us.UNCERTAINTY_SCENARIOS[POC_SCENARIO_KEY]
    assert "iDT1a" in cfg["label_a"]
    assert "iDT1b" in cfg["label_b"]


def test_poc_provider_a_qos_matches_cdt1_idt1a():
    cfg = us.UNCERTAINTY_SCENARIOS[POC_SCENARIO_KEY]
    pa = cfg["prov_a"]
    assert pa["acc_mean"] == pytest.approx(0.95, abs=1e-6)
    assert pa["lat_mean"] == pytest.approx(12.0, abs=1e-6)
    assert pa["rel_mean"] == pytest.approx(0.97, abs=1e-6)


def test_poc_provider_b_qos_matches_cdt1_idt1b():
    cfg = us.UNCERTAINTY_SCENARIOS[POC_SCENARIO_KEY]
    pb = cfg["prov_b"]
    assert pb["acc_mean"] == pytest.approx(0.88, abs=1e-6)
    assert pb["lat_mean"] == pytest.approx(18.0, abs=1e-6)
    assert pb["rel_mean"] == pytest.approx(0.93, abs=1e-6)


def test_poc_scenario_uncertainty_structure_preserved():
    cfg = us.UNCERTAINTY_SCENARIOS[POC_SCENARIO_KEY]
    assert cfg["prov_a"]["sigma_q"] > cfg["prov_b"]["sigma_q"]
    assert cfg["prov_a"]["sigma_a"] > cfg["prov_b"]["sigma_a"]


# ── Task C tests ──────────────────────────────────────────────────────────────

import pandas as pd


def test_risk_sweep_returns_dataframe():
    result = us.sweep_risk_factors(
        lambda_values=[1.0, 1.5],
        mu_values=[1.0, 1.5],
        scenario_key=POC_SCENARIO_KEY,
        runs=3,
        base_seed=42,
    )
    assert isinstance(result, pd.DataFrame)


def test_risk_sweep_columns():
    result = us.sweep_risk_factors(
        lambda_values=[1.0, 1.5],
        mu_values=[1.0, 1.5],
        scenario_key=POC_SCENARIO_KEY,
        runs=3,
        base_seed=42,
    )
    assert "lambda" in result.columns
    assert "mu" in result.columns
    assert "mean_utility_gap" in result.columns


def test_risk_sweep_shape():
    result = us.sweep_risk_factors(
        lambda_values=[0.5, 1.0, 1.5],
        mu_values=[0.5, 1.0, 1.5],
        scenario_key=POC_SCENARIO_KEY,
        runs=3,
        base_seed=42,
    )
    assert len(result) == 9  # 3x3 grid


def test_risk_sweep_gap_nonnegative():
    result = us.sweep_risk_factors(
        lambda_values=[0.5, 1.0, 1.5, 2.0],
        mu_values=[0.5, 1.0, 1.5, 2.0],
        scenario_key=POC_SCENARIO_KEY,
        runs=10,
        base_seed=42,
    )
    assert result["mean_utility_gap"].min() > -0.20


def test_risk_sweep_output_file(tmp_path):
    us.sweep_risk_factors(
        lambda_values=[1.0, 1.5],
        mu_values=[1.0, 1.5],
        scenario_key=POC_SCENARIO_KEY,
        runs=3,
        base_seed=42,
        output_csv=str(tmp_path / "sweep.csv"),
    )
    assert (tmp_path / "sweep.csv").exists()
    df = pd.read_csv(tmp_path / "sweep.csv")
    assert len(df) == 4
