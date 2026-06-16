#!/usr/bin/env python3
"""
uncertainty_sim.py — Uncertainty-Aware Selection Simulation
============================================================

Evaluates three selection policies under degradation and measurement uncertainty:

  1. baseline          — availability-based: switches when reliability < threshold
  2. qos_aware         — deterministic utility: selects by weighted QoS utility
  3. uncertainty_aware — risk-adjusted utility with SEPARATE quality uncertainty (σ_q)
                         and availability uncertainty (σ_a) terms

Three scenario presets with fully editable parameters:
  scenario_1: limited_uncertainty  — small σ values; QoS-aware ≈ uncertainty-aware
  scenario_2: clear_uncertainty    — noticeable σ separation; policies begin to diverge
  scenario_3: extreme_uncertainty  — very high σ for provider A; uncertainty-aware
                                     strongly prefers B even when A has better nominal QoS

Risk-adjusted utility (policy 3):
  adj_utility = w_acc * max(0, acc - risk_q * σ_q(t))
              + w_lat * lat_score
              + w_rel * max(0, rel - risk_a * σ_a(t))

  where σ_q(t) and σ_a(t) grow during degradation episodes.

Output layout:
  results/uncertainty_simulation/scenario_{1,2,3}/
    manifest.json
    data/aggregated.csv
    data/runs/run_NNNN_seed_SSSS.csv

  docs/figures/uncertainty_simulation/scenario_{1,2,3}/
    utility_over_time.{png,pdf}
    selection_over_time.{png,pdf}
    difference_plots.{png,pdf}
    uncertainty_evolution.{png,pdf}
    provider_scores.{png,pdf}
    summary.{png,pdf}

Usage:
  python scripts/uncertainty_sim.py
  python scripts/uncertainty_sim.py --scenario 2 --runs 50 --seed 42
  python scripts/uncertainty_sim.py --plot-only --scenario all
  python scripts/uncertainty_sim.py --output-dir results --figures-dir docs/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_LATENCY_MS = 100.0   # latency normalisation ceiling

POLICIES = ["baseline", "qos_aware", "uncertainty_aware"]

# Uncertainty sigma grows during degradation by this multiplier at hard-fail
SIGMA_DEGRADE_FACTOR_Q = 3.0   # quality uncertainty amplifier
SIGMA_DEGRADE_FACTOR_A = 2.0   # availability uncertainty amplifier

# ── Scenario configurations ───────────────────────────────────────────────────

UNCERTAINTY_SCENARIOS: dict[str, dict] = {

    # ── Scenario 1: Limited uncertainty ──────────────────────────────────────
    # Small σ for both providers. QoS-aware and uncertainty-aware behave similarly.
    # Baseline is visibly weaker during degradation.
    "scenario_1": {
        "label": "Limited Uncertainty",
        "description": (
            "Small measurement uncertainty for both providers. "
            "The QoS-aware and uncertainty-aware policies behave similarly because "
            "σ is small enough that risk adjustment barely changes rankings. "
            "The baseline is still visibly weaker during degradation episodes."
        ),
        "label_a": "Provider A",
        "label_b": "Provider B",
        # Provider A — quality sensor: higher accuracy, slower, higher σ
        "prov_a": {
            "acc_mean": 0.92, "lat_mean": 35.0, "rel_mean": 0.95,
            "sigma_q":  0.02,   # quality (accuracy) measurement uncertainty
            "sigma_a":  0.02,   # availability (reliability) measurement uncertainty
        },
        # Provider B — fast sensor: moderate accuracy, faster, lower σ
        "prov_b": {
            "acc_mean": 0.75, "lat_mean": 10.0, "rel_mean": 0.87,
            "sigma_q":  0.015,
            "sigma_a":  0.015,
        },
        "weights":           (0.45, 0.20, 0.35),  # (w_acc, w_lat, w_rel)
        "risk_q":            1.0,    # risk aversion for quality uncertainty
        "risk_a":            1.0,    # risk aversion for availability uncertainty
        "avail_threshold":   0.15,   # baseline switches when rel < this
        "util_threshold":    0.55,   # qos_aware emergency switch
        "hysteresis":        0.06,   # proactive switch-back threshold
        # Degradation (2 episodes)
        "sim_duration_s":    120,
        "recovery_s":        10,
        "episode_count":     2,
        "degrade_rate":      (0.04, 0.10),
        "onset_range":       (15.0, 25.0),
        "degrade_window":    ( 8.0, 12.0),
        "fail_window":       ( 6.0, 10.0),
        "inter_gap":         (12.0, 20.0),
    },

    # ── Scenario 2: Clear uncertainty ─────────────────────────────────────────
    # Noticeable σ separation. Provider A has clearly higher uncertainty.
    # P3 begins to diverge from P2; in some timesteps P3 prefers B over A.
    "scenario_2": {
        "label": "Clear Uncertainty",
        "description": (
            "Noticeable uncertainty separation: provider A has clearly higher σ_q and σ_a "
            "than provider B. The uncertainty-aware policy begins to diverge from the "
            "deterministic QoS-aware policy. At nominal QoS the two policies may still "
            "agree, but during degradation P3 switches away from A earlier and more reliably."
        ),
        "label_a": "Provider A",
        "label_b": "Provider B",
        "prov_a": {
            "acc_mean": 0.92, "lat_mean": 35.0, "rel_mean": 0.95,
            "sigma_q":  0.08,
            "sigma_a":  0.06,
        },
        "prov_b": {
            "acc_mean": 0.75, "lat_mean": 10.0, "rel_mean": 0.87,
            "sigma_q":  0.02,
            "sigma_a":  0.02,
        },
        "weights":          (0.45, 0.20, 0.35),
        "risk_q":           1.5,
        "risk_a":           1.5,
        "avail_threshold":  0.15,
        "util_threshold":   0.55,
        "hysteresis":       0.05,
        "sim_duration_s":   120,
        "recovery_s":       10,
        "episode_count":    2,
        "degrade_rate":     (0.05, 0.12),
        "onset_range":      (15.0, 25.0),
        "degrade_window":   ( 8.0, 14.0),
        "fail_window":      ( 6.0, 12.0),
        "inter_gap":        (12.0, 20.0),
    },

    # ── Scenario 3: Extreme uncertainty ──────────────────────────────────────
    # Very high σ for provider A. P3 strongly prefers B even at nominal QoS.
    # This clearly shows that high nominal QoS is not always sufficient.
    #
    # Nominal P2 utility:  A=0.895  B=0.818  → P2 prefers A  (gap=0.077)
    # Risk-adjusted P3:    A=0.649  B=0.786  → P3 prefers B  (gap=0.137)  ← key result
    "scenario_3": {
        "label": "Extreme Uncertainty",
        "description": (
            "Very high uncertainty for provider A (σ_q=0.18, σ_a=0.12) with strong risk "
            "aversion (ρ=2.0). Provider A has better nominal QoS, but its risk-adjusted "
            "utility is substantially lower. The uncertainty-aware policy clearly prefers "
            "provider B throughout the simulation, while the QoS-aware policy prefers A. "
            "Three degradation episodes make the relative advantage unmistakable."
        ),
        "label_a": "Provider A",
        "label_b": "Provider B",
        "prov_a": {
            "acc_mean": 0.93, "lat_mean": 30.0, "rel_mean": 0.96,
            "sigma_q":  0.18,
            "sigma_a":  0.12,
        },
        "prov_b": {
            "acc_mean": 0.74, "lat_mean":  8.0, "rel_mean": 0.86,
            "sigma_q":  0.02,
            "sigma_a":  0.02,
        },
        "weights":          (0.45, 0.20, 0.35),
        "risk_q":           2.0,
        "risk_a":           2.0,
        "avail_threshold":  0.15,
        "util_threshold":   0.50,
        "hysteresis":       0.04,
        "sim_duration_s":   150,
        "recovery_s":       10,
        "episode_count":    3,
        "degrade_rate":     (0.06, 0.15),
        "onset_range":      (15.0, 25.0),
        "degrade_window":   ( 8.0, 14.0),
        "fail_window":      ( 6.0, 12.0),
        "inter_gap":        (10.0, 18.0),
    },

    "scenario_2_idt": {
        "label": "PoC-Grounded (iDT1a vs. iDT1b)",
        "description": (
            "QoS values derived from cDT1 ProviderConfig in the PoC Go implementation. "
            "iDT1a is the primary inspection-robot provider; iDT1b is the fallback."
        ),
        "label_a": "iDT1a (primary, inspection robot)",
        "label_b": "iDT1b (fallback, inspection robot)",
        "prov_a": {"acc_mean": 0.95, "lat_mean": 12.0, "rel_mean": 0.97,
                   "sigma_q": 0.08, "sigma_a": 0.06},
        "prov_b": {"acc_mean": 0.88, "lat_mean": 18.0, "rel_mean": 0.93,
                   "sigma_q": 0.02, "sigma_a": 0.02},
        "weights":            (0.45, 0.20, 0.35),
        "risk_q":             1.5,
        "risk_a":             1.5,
        "avail_threshold":    0.15,
        "util_threshold":     0.55,
        "hysteresis":         0.05,
        "sim_duration_s":     120,
        "recovery_s":         10,
        "episode_count":      2,
        "degrade_rate":       (0.05, 0.12),
        "onset_range":        (15.0, 25.0),
        "degrade_window":     (8.0, 14.0),
        "fail_window":        (6.0, 12.0),
        "inter_gap":          (12.0, 20.0),
    },
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def compute_utility(acc: float, lat_ms: float, rel: float,
                    w_acc: float, w_lat: float, w_rel: float) -> float:
    """Weighted additive utility in [0, 1].  Latency is inverted."""
    lat_score = 1.0 - min(1.0, lat_ms / MAX_LATENCY_MS)
    return w_acc * acc + w_lat * lat_score + w_rel * rel


def compute_risk_adjusted_utility(acc: float, lat_ms: float, rel: float,
                                   sigma_q: float, sigma_a: float,
                                   risk_q: float, risk_a: float,
                                   w_acc: float, w_lat: float, w_rel: float) -> float:
    """Risk-adjusted utility using separate quality and availability uncertainty.

    Quality uncertainty (sigma_q) penalises the accuracy component.
    Availability uncertainty (sigma_a) penalises the reliability component.
    Latency is treated deterministically (no latency uncertainty term).

    Formula:
        adj_acc = max(0, acc - risk_q * sigma_q)
        adj_rel = max(0, rel - risk_a * sigma_a)
        adj_utility = w_acc * adj_acc + w_lat * lat_score + w_rel * adj_rel
    """
    lat_score = 1.0 - min(1.0, lat_ms / MAX_LATENCY_MS)
    adj_acc = max(0.0, acc - risk_q * sigma_q)
    adj_rel = max(0.0, rel - risk_a * sigma_a)
    return w_acc * adj_acc + w_lat * lat_score + w_rel * adj_rel


# ── Degradation model ─────────────────────────────────────────────────────────

def _degradation_factor(t: float, ep: dict) -> float:
    """Return normalised degradation intensity in [0, 1] for a single episode.

    0 = healthy, 1 = hard failure.
    """
    onset   = ep["onset_s"]
    fail_at = ep["fail_at_s"]
    rec_at  = ep["recover_at_s"]
    rec_end = rec_at + ep["recovery_s"]

    if t < onset or t >= rec_end:
        return 0.0
    if onset <= t < fail_at:
        return min(1.0, (t - onset) / max(1e-9, fail_at - onset))
    if fail_at <= t < rec_at:
        return 1.0
    # Recovery phase
    return max(0.0, 1.0 - (t - rec_at) / max(1e-9, rec_end - rec_at))


def _provider_qos_at(t: float, nom: dict, episodes: list, recovery_s: float) -> dict:
    """Return live (true) QoS of a provider at time t."""
    for ep in episodes:
        onset   = ep["onset_s"]
        rate    = ep["rate"]
        fail_at = ep["fail_at_s"]
        rec_at  = ep["recover_at_s"]
        rec_end = rec_at + recovery_s

        if onset <= t < fail_at:
            e = t - onset
            return {
                "acc":    max(0.0, nom["acc_mean"]    - e * rate),
                "lat_ms": nom["lat_mean"] * (1.0 + e * rate * 2.0),
                "rel":    max(0.0, nom["rel_mean"]    - e * rate * 0.8),
            }
        if fail_at <= t < rec_at:
            return {"acc": 0.0, "lat_ms": nom["lat_mean"] * 5.0, "rel": 0.0}
        if rec_at <= t < rec_end:
            frac = (t - rec_at) / max(1e-9, recovery_s)
            return {
                "acc":    nom["acc_mean"]    * frac,
                "lat_ms": nom["lat_mean"]    * (4.0 - 3.0 * frac),
                "rel":    nom["rel_mean"]    * frac,
            }

    return {"acc": nom["acc_mean"], "lat_ms": nom["lat_mean"], "rel": nom["rel_mean"]}


def _sigma_at(t: float, sigma_base: float, episodes: list,
              recovery_s: float, factor: float) -> float:
    """Return dynamic uncertainty sigma at time t.

    Uncertainty grows proportionally to degradation severity:
        sigma(t) = sigma_base * (1 + factor * max_deg_factor(t))
    """
    max_deg = max((_degradation_factor(t, ep) for ep in episodes), default=0.0)
    return sigma_base * (1.0 + factor * max_deg)


# ── Policy implementations ────────────────────────────────────────────────────

def _select_baseline(active: str, qos: dict[str, dict], avail_threshold: float) -> str:
    """Policy 1: availability-based selection.

    Switches away from active provider only when its reliability falls below
    avail_threshold. Does not use quality values in the decision.
    Never proactively switches back.
    """
    if qos[active]["rel"] < avail_threshold:
        for alt, q in qos.items():
            if alt != active and q["rel"] >= avail_threshold:
                return alt
    return active


def _select_qos_aware(active: str, qos: dict[str, dict],
                       w_acc: float, w_lat: float, w_rel: float,
                       util_threshold: float, hysteresis: float) -> str:
    """Policy 2: deterministic QoS-aware selection.

    Emergency switch: active utility < util_threshold.
    Proactive switch: best alternative is better by > hysteresis.
    """
    aq = qos[active]
    u_active = compute_utility(aq["acc"], aq["lat_ms"], aq["rel"], w_acc, w_lat, w_rel)

    alternatives = {p: compute_utility(q["acc"], q["lat_ms"], q["rel"], w_acc, w_lat, w_rel)
                    for p, q in qos.items() if p != active}
    if not alternatives:
        return active

    best_alt, u_best = max(alternatives.items(), key=lambda x: x[1])
    should_switch = u_active < util_threshold or u_best > u_active + hysteresis
    if should_switch and u_best > u_active:
        return best_alt
    return active


def _select_uncertainty_aware(active: str, qos: dict[str, dict],
                               sigma_q: dict[str, float], sigma_a: dict[str, float],
                               risk_q: float, risk_a: float,
                               w_acc: float, w_lat: float, w_rel: float,
                               util_threshold: float, hysteresis: float) -> str:
    """Policy 3: QoS-aware selection with separate uncertainty terms.

    Quality uncertainty (sigma_q) and availability uncertainty (sigma_a) are
    handled independently. The risk-adjusted utility discounts providers whose
    measurements are highly uncertain, even if their nominal QoS looks attractive.

    Emergency switch: risk-adjusted utility of active < util_threshold.
    Proactive switch: best alternative's adj_utility > active's by > hysteresis.
    """
    aq = qos[active]
    u_active = compute_risk_adjusted_utility(
        aq["acc"], aq["lat_ms"], aq["rel"],
        sigma_q[active], sigma_a[active],
        risk_q, risk_a,
        w_acc, w_lat, w_rel,
    )

    alternatives = {}
    for p, q in qos.items():
        if p != active:
            alternatives[p] = compute_risk_adjusted_utility(
                q["acc"], q["lat_ms"], q["rel"],
                sigma_q[p], sigma_a[p],
                risk_q, risk_a,
                w_acc, w_lat, w_rel,
            )

    if not alternatives:
        return active

    best_alt, u_best = max(alternatives.items(), key=lambda x: x[1])
    should_switch = u_active < util_threshold or u_best > u_active + hysteresis
    if should_switch and u_best > u_active:
        return best_alt
    return active


# ── Episode builder ───────────────────────────────────────────────────────────

def _build_episodes(rng: np.random.Generator, providers: list[str],
                    cfg: dict) -> dict[str, list]:
    """Build degradation episode schedule for all providers.

    First episode: random provider; subsequent episodes alternate.
    Each episode covers: [onset_s, fail_at_s) gradual → [fail_at_s, recover_at_s) fail
    → [recover_at_s, recover_at_s + recovery_s) recovery.
    """
    ep_map: dict[str, list] = {p: [] for p in providers}
    current_start = 0.0
    degraded = str(rng.choice(providers))

    for i in range(cfg["episode_count"]):
        if i == 0:
            onset_s = float(rng.uniform(*cfg["onset_range"]))
        else:
            prev = ep_map[degraded][-1] if ep_map[degraded] else ep_map[
                [p for p in providers if p != degraded][0]][-1]
            rec_end = prev["recover_at_s"] + cfg["recovery_s"]
            onset_s = rec_end + float(rng.uniform(*cfg["inter_gap"]))
            # Alternate provider
            degraded = [p for p in providers if p != degraded][0]

        rate         = float(rng.uniform(*cfg["degrade_rate"]))
        fail_at_s    = onset_s + float(rng.uniform(*cfg["degrade_window"]))
        recover_at_s = fail_at_s + float(rng.uniform(*cfg["fail_window"]))

        ep = {
            "onset_s":      onset_s,
            "rate":         rate,
            "fail_at_s":    fail_at_s,
            "recover_at_s": recover_at_s,
            "recovery_s":   cfg["recovery_s"],
        }
        ep_map[degraded].append(ep)
        current_start = recover_at_s + cfg["recovery_s"]

    return ep_map


# ── Single run simulation ─────────────────────────────────────────────────────

def _run_single(seed: int, run_idx: int, cfg: dict) -> list[dict]:
    """Simulate all three policies for one run with the given seed.

    Returns a list of row dicts (one per timestep per policy).
    """
    rng  = np.random.default_rng(seed)

    providers = ["prov_a", "prov_b"]
    nom = {"prov_a": cfg["prov_a"], "prov_b": cfg["prov_b"]}

    ep_map = _build_episodes(rng, providers, cfg)

    w_acc, w_lat, w_rel = cfg["weights"]
    risk_q     = cfg["risk_q"]
    risk_a     = cfg["risk_a"]
    avail_thr  = cfg["avail_threshold"]
    util_thr   = cfg["util_threshold"]
    hysteresis = cfg["hysteresis"]

    # Initial provider selection (all start on prov_a)
    active = {pol: "prov_a" for pol in POLICIES}

    rows: list[dict] = []
    sim_dur = cfg["sim_duration_s"]
    rec_s   = cfg["recovery_s"]

    for t_int in range(sim_dur + 1):
        t = float(t_int)

        # True QoS
        true_qos: dict[str, dict] = {
            p: _provider_qos_at(t, nom[p], ep_map[p], rec_s) for p in providers
        }

        # Measurement noise (Box-Muller for normal sampling)
        def sample_normal(mean: float, std: float) -> float:
            u1, u2 = float(rng.random()), float(rng.random())
            z = np.sqrt(-2 * np.log(max(1e-10, u1))) * np.cos(2 * np.pi * u2)
            return float(mean + std * z)

        # Dynamic sigma (grows with degradation severity)
        sigma_q: dict[str, float] = {}
        sigma_a: dict[str, float] = {}
        for p in providers:
            sigma_q[p] = _sigma_at(t, nom[p]["sigma_q"], ep_map[p], rec_s, SIGMA_DEGRADE_FACTOR_Q)
            sigma_a[p] = _sigma_at(t, nom[p]["sigma_a"], ep_map[p], rec_s, SIGMA_DEGRADE_FACTOR_A)

        # Noisy measurements (all policies see the same noise)
        meas_qos: dict[str, dict] = {}
        for p in providers:
            meas_qos[p] = {
                "acc":    max(0.0, min(1.0, sample_normal(true_qos[p]["acc"],    sigma_q[p]))),
                "lat_ms": max(0.0,           sample_normal(true_qos[p]["lat_ms"], 0.0)),
                "rel":    max(0.0, min(1.0, sample_normal(true_qos[p]["rel"],    sigma_a[p]))),
            }

        # Policy decisions
        prev_active = dict(active)

        active["baseline"] = _select_baseline(
            active["baseline"], meas_qos, avail_thr)

        active["qos_aware"] = _select_qos_aware(
            active["qos_aware"], meas_qos, w_acc, w_lat, w_rel, util_thr, hysteresis)

        active["uncertainty_aware"] = _select_uncertainty_aware(
            active["uncertainty_aware"], meas_qos, sigma_q, sigma_a,
            risk_q, risk_a, w_acc, w_lat, w_rel, util_thr, hysteresis)

        # Utilities of both providers at this timestep (policy-independent)
        tq_a = true_qos["prov_a"]
        tq_b = true_qos["prov_b"]
        utility_a = compute_utility(tq_a["acc"], tq_a["lat_ms"], tq_a["rel"],
                                    w_acc, w_lat, w_rel)
        utility_b = compute_utility(tq_b["acc"], tq_b["lat_ms"], tq_b["rel"],
                                    w_acc, w_lat, w_rel)

        # True utility of selected provider for each policy
        for pol in POLICIES:
            sel = active[pol]
            tq  = true_qos[sel]
            true_util = compute_utility(tq["acc"], tq["lat_ms"], tq["rel"],
                                        w_acc, w_lat, w_rel)
            rows.append({
                "run":              run_idx,
                "seed":             seed,
                "t":                t_int,
                "policy":           pol,
                "selected":         sel,
                "true_acc":         round(tq["acc"],    6),
                "true_lat_ms":      round(tq["lat_ms"], 4),
                "true_rel":         round(tq["rel"],    6),
                "true_utility":     round(true_util,    6),
                "sigma_q_selected": round(sigma_q[sel], 6),
                "sigma_a_selected": round(sigma_a[sel], 6),
                "sigma_q_A":        round(sigma_q["prov_a"], 6),
                "sigma_q_B":        round(sigma_q["prov_b"], 6),
                "sigma_a_A":        round(sigma_a["prov_a"], 6),
                "sigma_a_B":        round(sigma_a["prov_b"], 6),
                "failover_event":   1 if active[pol] != prev_active[pol] else 0,
                "utility_a":        round(utility_a, 6),
                "utility_b":        round(utility_b, 6),
            })

    return rows


# ── Main simulation driver ────────────────────────────────────────────────────

def run_scenario(scenario_key: str, runs: int, base_seed: int,
                 output_dir: Path, cfg: dict) -> None:
    """Run the full simulation for one scenario and write CSV output."""
    run_dir  = output_dir / "data" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "data" / "aggregated.csv"

    header = [
        "run", "seed", "t", "policy", "selected",
        "true_acc", "true_lat_ms", "true_rel", "true_utility",
        "sigma_q_selected", "sigma_a_selected",
        "sigma_q_A", "sigma_q_B", "sigma_a_A", "sigma_a_B",
        "failover_event",
        "utility_a", "utility_b",
    ]

    all_rows: list[dict] = []

    for run in range(runs):
        seed = base_seed + run
        rows = _run_single(seed, run, cfg)
        all_rows.extend(rows)

        csv_path = run_dir / f"run_{run:04d}_seed_{seed}.csv"
        _write_csv(csv_path, header, rows)
        print(f"  [{scenario_key}] run={run:3d}  seed={seed}", flush=True)

    _write_csv(agg_path, header, all_rows)
    print(f"  [{scenario_key}] aggregated ({len(all_rows)} rows) → {agg_path}", flush=True)

    # Manifest
    manifest = {
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario":        scenario_key,
        "label":           cfg["label"],
        "description":     cfg["description"],
        "base_seed":       base_seed,
        "runs":            runs,
        "policies":        POLICIES,
        "provider_a":      cfg["prov_a"],
        "provider_b":      cfg["prov_b"],
        "weights":         {"w_acc": cfg["weights"][0], "w_lat": cfg["weights"][1],
                            "w_rel": cfg["weights"][2]},
        "risk_q":          cfg["risk_q"],
        "risk_a":          cfg["risk_a"],
        "avail_threshold": cfg["avail_threshold"],
        "util_threshold":  cfg["util_threshold"],
        "hysteresis":      cfg["hysteresis"],
        "sim_duration_s":  cfg["sim_duration_s"],
        "recovery_s":      cfg["recovery_s"],
        "episode_count":   cfg["episode_count"],
    }
    mpath = output_dir / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest → {mpath}", flush=True)


# ── Plotting ──────────────────────────────────────────────────────────────────
#
# Figures are sized for IEEE double-column format: one figure per column.
# IEEE single-column width = 3.5 in (88.9 mm).
# All font sizes, line widths, and marker sizes are chosen so that the text
# is readable at that physical size when the PDF is embedded in a paper.

# Publication colour scheme (consistent with existing plot.py)
C_BASELINE  = "#b91c1c"   # dark red
C_QOS       = "#1d4ed8"   # dark blue
C_UNCERT    = "#15803d"   # dark green
C_PROV_A    = "#7c3aed"   # purple
C_PROV_B    = "#d97706"   # amber
C_DEGRADE   = "#d1d5db"   # light gray

# IEEE single-column figure dimensions
COL_W  = 3.5    # inches — one column of an IEEE double-column paper
ROW_H  = 2.4    # inches — default panel height
FIG_DPI = 300   # 300 dpi for print quality

# Typography — sized to be legible at 3.5 in column width
FS_TICK    = 8    # axis tick labels
FS_LABEL   = 9    # axis labels (xlabel / ylabel)
FS_TITLE   = 9    # subplot titles
FS_LEGEND  = 7.5  # legend entries
FS_ANNOT   = 7    # minor annotations

# Line and marker weights
LW        = 1.5   # main median curve
LW_BAND   = 0.8   # p10/p90 dashed lines
LW_GRID   = 0.4
BAND_ALPHA = 0.15
MARKER_EVERY = 10
MS = 4            # marker size

# Apply globally so every plt.subplots() call inherits these defaults
plt.rcParams.update({
    "font.size":        FS_TICK,
    "axes.titlesize":   FS_TITLE,
    "axes.labelsize":   FS_LABEL,
    "xtick.labelsize":  FS_TICK,
    "ytick.labelsize":  FS_TICK,
    "legend.fontsize":  FS_LEGEND,
    "figure.dpi":       FIG_DPI,
    "axes.linewidth":   0.6,
    "lines.linewidth":  LW,
    "patch.linewidth":  0.5,
})

POLICY_STYLE: dict[str, tuple] = {
    "baseline":          (C_BASELINE, "--", "s", "Baseline"),
    "qos_aware":         (C_QOS,      "-",  "o", "QoS-aware"),
    "uncertainty_aware": (C_UNCERT,   "-.", "^", "Uncertainty-aware"),
}


def _style(ax, ylim=None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=LW_GRID, alpha=0.6, color="#9ca3af")
    ax.tick_params(labelsize=FS_TICK, pad=2)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = directory / f"{name}.{ext}"
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  saved → {out}")
    plt.close(fig)


def _plot_provider_comparison(data_dir: Path, fig_dir: Path, scenario_key: str,
                               cfg: dict, util_by: dict, util_a_by: dict,
                               util_b_by: dict, ts: list) -> None:
    """Plot iDT-level and cDT-level utility to show cascade effect."""
    label_a = cfg.get("label_a", "Provider A")
    label_b = cfg.get("label_b", "Provider B")

    fig, ax = plt.subplots(figsize=(COL_W, ROW_H * 1.5))

    x, mA, p10A, p90A = _band(util_a_by, ts)
    x, mB, p10B, p90B = _band(util_b_by, ts)
    _plot_band(ax, x, mA, p10A, p90A, C_PROV_A, "-", None,
               label_med=f"{label_a} utility")
    _plot_band(ax, x, mB, p10B, p90B, C_PROV_B, "--", None,
               label_med=f"{label_b} utility")

    x, mQ, p10Q, p90Q = _band(util_by["qos_aware"], ts)
    x, mBL, p10BL, p90BL = _band(util_by["baseline"], ts)
    _plot_band(ax, x, mQ, p10Q, p90Q, C_QOS, "-.", "o",
               label_med="cDT1 under QoS-aware")
    _plot_band(ax, x, mBL, p10BL, p90BL, C_BASELINE, ":", "s",
               label_med="cDT1 under availability-based")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Utility")
    ax.set_title(f"{cfg['label']} — provider utility cascade\n"
                 f"(iDT degradation → cDT output; shaded p10–p90)")
    ax.set_xlim(0, cfg["sim_duration_s"])
    _style(ax, ylim=(-0.02, 1.12))
    ax.legend(fontsize=FS_LEGEND, loc="lower left", framealpha=0.92, ncol=1)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "provider_comparison")


def sweep_risk_factors(lambda_values, mu_values, scenario_key="scenario_2_idt",
                       runs=30, base_seed=42, output_csv=None):
    """
    Sweep lambda (risk_q) and mu (risk_a) over given value lists.
    Returns a DataFrame with columns: lambda, mu, mean_utility_gap, std_utility_gap.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. pip install pandas")

    base_cfg = UNCERTAINTY_SCENARIOS[scenario_key]
    records = []
    for lam in lambda_values:
        for mu in mu_values:
            cfg = {**base_cfg, "risk_q": lam, "risk_a": mu}
            gaps = []
            for run_idx in range(runs):
                seed = base_seed + run_idx
                rows = _run_single(seed, run_idx, cfg)
                df = pd.DataFrame(rows)
                sigma_q_base = base_cfg["prov_a"]["sigma_q"]
                deg_mask = df["sigma_q_A"] > sigma_q_base * 1.5
                if deg_mask.sum() == 0:
                    continue
                ua_util = df.loc[deg_mask & (df["policy"] == "uncertainty_aware"),
                                 "true_utility"]
                qa_util = df.loc[deg_mask & (df["policy"] == "qos_aware"),
                                 "true_utility"]
                if len(ua_util) > 0 and len(qa_util) > 0:
                    gaps.append(float(ua_util.mean()) - float(qa_util.mean()))
            gap_series = pd.Series(gaps) if gaps else pd.Series([0.0])
            records.append({
                "lambda": lam,
                "mu": mu,
                "mean_utility_gap": float(gap_series.mean()),
                "std_utility_gap":  float(gap_series.std()) if len(gaps) > 1 else 0.0,
            })
    result = pd.DataFrame(records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def plot_risk_factor_heatmap(df, fig_dir) -> None:
    """Render an annotated heatmap of mean_utility_gap over (lambda, mu) grid."""
    if not _HAS_MPL:
        print("  [skip] matplotlib not available", file=sys.stderr)
        return

    fig_dir = Path(fig_dir)
    lambdas = sorted(df["lambda"].unique())
    mus = sorted(df["mu"].unique())
    matrix = np.zeros((len(mus), len(lambdas)))
    for i, mu in enumerate(mus):
        for j, lam in enumerate(lambdas):
            val = df.loc[(df["lambda"] == lam) & (df["mu"] == mu), "mean_utility_gap"]
            matrix[i, j] = float(val.iloc[0]) if len(val) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(COL_W * 1.1, ROW_H * 1.4))
    im = ax.imshow(matrix, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=matrix.min(), vmax=matrix.max())
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{v:.1f}" for v in lambdas], fontsize=FS_TICK)
    ax.set_yticks(range(len(mus)))
    ax.set_yticklabels([f"{v:.1f}" for v in mus], fontsize=FS_TICK)
    ax.set_xlabel(r"$\lambda$ (risk$_q$)")
    ax.set_ylabel(r"$\mu$ (risk$_a$)")
    ax.set_title(r"Utility gap: uncertainty-aware $-$ QoS-aware" "\n"
                 r"(during degradation windows; green = uncertainty-aware wins)")
    for i in range(len(mus)):
        for j in range(len(lambdas)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=FS_ANNOT, color="black")
    plt.colorbar(im, ax=ax, label="Mean utility gap", shrink=0.85)
    _style(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "risk_factor_heatmap")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _band(data_by_t: dict, ts: list) -> tuple:
    """Return (x, median, p10, p90) arrays."""
    meds, p10s, p90s = [], [], []
    for t in ts:
        v = np.array(data_by_t[t], dtype=float)
        meds.append(float(np.median(v)))
        p10s.append(float(np.percentile(v, 10)))
        p90s.append(float(np.percentile(v, 90)))
    return (np.array(ts, dtype=float),
            np.array(meds), np.array(p10s), np.array(p90s))


def _plot_band(ax, x, med, p10, p90, color, ls="-", marker=None,
               label_med=None, label_band=None) -> None:
    kw: dict = dict(color=color, linewidth=LW, linestyle=ls, zorder=3)
    if marker:
        step = max(1, len(x) // MARKER_EVERY)
        kw.update(marker=marker, markevery=step, markersize=MS, markeredgewidth=0.4)
    ax.plot(x, med, label=label_med, **kw)
    bkw = dict(color=color, linewidth=LW_BAND, linestyle="--", alpha=0.70, zorder=2)
    ax.plot(x, p10, **bkw)
    ax.plot(x, p90, label=label_band, **bkw)
    ax.fill_between(x, p10, p90, color=color, alpha=BAND_ALPHA, zorder=1)


def _generate_plots(data_dir: Path, fig_dir: Path, scenario_key: str, cfg: dict) -> None:
    """Generate all six plots for one scenario."""
    if not _HAS_MPL:
        print("  [skip] matplotlib not available", file=sys.stderr)
        return

    agg_path = data_dir / "data" / "aggregated.csv"
    if not agg_path.exists():
        print(f"  [skip] {agg_path} not found")
        return

    rows  = _read_csv(agg_path)
    label = cfg["label"]
    sim_t = cfg["sim_duration_s"]
    n_runs = len({r["run"] for r in rows if r["policy"] == "baseline"})

    # Organise data: util_by[policy][t] = [utility values across runs]
    util_by:  dict[str, dict[int, list]] = {p: defaultdict(list) for p in POLICIES}
    sel_by:   dict[str, dict[int, list]] = {p: defaultdict(list) for p in POLICIES}
    sq_A_by:  dict[int, list] = defaultdict(list)
    sq_B_by:  dict[int, list] = defaultdict(list)
    sa_A_by:  dict[int, list] = defaultdict(list)
    sa_B_by:  dict[int, list] = defaultdict(list)
    util_a_by: dict[int, list] = defaultdict(list)
    util_b_by: dict[int, list] = defaultdict(list)

    for r in rows:
        pol = r["policy"]
        t   = int(r["t"])
        util_by[pol][t].append(float(r["true_utility"]))
        # Selection: 0=prov_a, 1=prov_b
        sel_by[pol][t].append(1.0 if r["selected"] == "prov_b" else 0.0)
        if pol == "baseline":
            sq_A_by[t].append(float(r["sigma_q_A"]))
            sq_B_by[t].append(float(r["sigma_q_B"]))
            sa_A_by[t].append(float(r["sigma_a_A"]))
            sa_B_by[t].append(float(r["sigma_a_B"]))
            if "utility_a" in r and r["utility_a"]:
                util_a_by[t].append(float(r["utility_a"]))
            if "utility_b" in r and r["utility_b"]:
                util_b_by[t].append(float(r["utility_b"]))

    ts = sorted(util_by["baseline"].keys())

    # ── Plot 1: Utility over time ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(COL_W, ROW_H * 1.3))
    for pol in POLICIES:
        c, ls, mk, name = POLICY_STYLE[pol]
        x, med, p10, p90 = _band(util_by[pol], ts)
        _plot_band(ax, x, med, p10, p90, c, ls, mk,
                   label_med=f"{name} (med.)",
                   label_band=f"{name} (p10/p90)")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Utility")
    ax.set_title(f"{label} — utility over time\n"
                 f"({n_runs} runs, shaded p10–p90)")
    ax.set_xlim(0, sim_t)
    _style(ax, ylim=(-0.02, 1.12))
    _legend(ax, POLICIES)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "utility_over_time")

    # ── Plot 2: Selection over time ───────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, ROW_H * 2.5), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    for ax, pol in zip(axes, POLICIES):
        c, ls, mk, name = POLICY_STYLE[pol]
        x, med, p10, p90 = _band(sel_by[pol], ts)
        ax.fill_between(x, p10, p90, color=c, alpha=0.18)
        ax.plot(x, med, color=c, linewidth=LW,
                label=f"{name}: frac. using B")
        ax.axhline(0.5, color="#9ca3af", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Frac. → B")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontweight="bold")
        _style(ax)
        ax.legend(loc="upper right", handlelength=1.5)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(f"{label} — provider selection\n"
                 "(1.0 = all runs chose B, 0.0 = all chose A)",
                 fontsize=FS_TITLE)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "selection_over_time")

    # ── Plot 3: Difference plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, ROW_H * 2.7), sharex=True,
                             gridspec_kw={"hspace": 0.40})

    diff_pairs = [
        ("qos_aware",         "baseline",  "QoS-aware − Baseline (P2−P1)",
         C_QOS,    "P2−P1"),
        ("uncertainty_aware", "qos_aware", "Uncertainty-aware − QoS-aware (P3−P2)",
         C_UNCERT, "P3−P2"),
        ("uncertainty_aware", "baseline",  "Uncertainty-aware − Baseline (P3−P1)",
         "#7c3aed", "P3−P1"),
    ]

    for ax, (polA, polB, title, color, short) in zip(axes, diff_pairs):
        diff_by_t: dict[int, list] = defaultdict(list)
        for t in ts:
            ua = np.array(sorted(util_by[polA][t]))
            ub = np.array(sorted(util_by[polB][t]))
            n  = min(len(ua), len(ub))
            for d in (ua[:n] - ub[:n]):
                diff_by_t[t].append(float(d))
        x, med, p10, p90 = _band(diff_by_t, ts)
        ax.fill_between(x, p10, p90, color=color, alpha=0.18, zorder=1)
        ax.plot(x, p10, color=color, linewidth=LW_BAND, linestyle="--", alpha=0.70, zorder=2)
        ax.plot(x, p90, color=color, linewidth=LW_BAND, linestyle="--", alpha=0.70, zorder=2)
        ax.plot(x, med, color=color, linewidth=LW, zorder=3,
                label=f"Median ({short})")
        ax.axhline(0, color="#374151", linewidth=0.8, linestyle="--", label="No advantage")
        ax.set_ylabel("Advantage")
        ax.set_title(title, fontweight="bold")
        _style(ax)
        ax.legend(handlelength=1.5)

    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(f"{label} — utility differences\n"
                 "(positive = first policy wins; shaded p10–p90)",
                 fontsize=FS_TITLE)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "difference_plots")

    # ── Plot 4: Uncertainty evolution ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, ROW_H * 2.0), sharex=True,
                                    gridspec_kw={"hspace": 0.35})
    for ax, (d_A, d_B, ylabel, title_sfx) in zip(
            [ax1, ax2],
            [(sq_A_by, sq_B_by, r"$\sigma_q$", r"Quality uncertainty $\sigma_q$"),
             (sa_A_by, sa_B_by, r"$\sigma_a$", r"Availability uncertainty $\sigma_a$")]):
        x, mA, _, _ = _band(d_A, ts)
        x, mB, _, _ = _band(d_B, ts)
        ax.plot(x, mA, color=C_PROV_A, linewidth=LW,
                label=cfg.get("label_a", "Provider A"))
        ax.plot(x, mB, color=C_PROV_B, linewidth=LW, linestyle="--",
                label=cfg.get("label_b", "Provider B"))
        ax.set_ylabel(ylabel)
        ax.set_title(title_sfx, fontweight="bold")
        _style(ax)
        ax.legend(handlelength=1.5)
    ax2.set_xlabel("Simulation time (s)")
    fig.suptitle(f"{label} — uncertainty evolution\n"
                 r"($\sigma$ grows during degradation)",
                 fontsize=FS_TITLE)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "uncertainty_evolution")

    # ── Plot 5: Provider scores at representative moments ────────────────────
    w_acc, w_lat, w_rel = cfg["weights"]
    nom_a = cfg["prov_a"]
    nom_b = cfg["prov_b"]
    sim_dur_s = cfg["sim_duration_s"]
    t_reps = [int(sim_dur_s * 0.15), int(sim_dur_s * 0.40),
              int(sim_dur_s * 0.65), int(sim_dur_s * 0.90)]

    fig, ax = plt.subplots(figsize=(COL_W, ROW_H * 1.4))
    bar_w = 0.16
    n_groups = len(t_reps)
    x_base = np.arange(n_groups)
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]
    bar_cfgs = [
        ("P2: A (nom.)",  C_QOS,    "//",   lambda t: _nominal_utility(nom_a, w_acc, w_lat, w_rel)),
        ("P2: B (nom.)",  C_QOS,    "\\\\", lambda t: _nominal_utility(nom_b, w_acc, w_lat, w_rel)),
        ("P3: A (adj.)",  C_UNCERT, "//",   lambda t: _risk_adj_utility(
            nom_a, cfg["risk_q"], cfg["risk_a"], w_acc, w_lat, w_rel,
            nom_a["sigma_q"], nom_a["sigma_a"])),
        ("P3: B (adj.)",  C_UNCERT, "\\\\", lambda t: _risk_adj_utility(
            nom_b, cfg["risk_q"], cfg["risk_a"], w_acc, w_lat, w_rel,
            nom_b["sigma_q"], nom_b["sigma_a"])),
    ]
    for off, (blabel, bcolor, bhatch, bfn) in zip(offsets, bar_cfgs):
        vals = [bfn(t) for t in t_reps]
        ax.bar(x_base + off, vals, bar_w, label=blabel,
               color=bcolor, hatch=bhatch, alpha=0.78,
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"t={t}s" for t in t_reps])
    ax.set_ylabel("Utility / adj. utility")
    ax.set_title(f"{label} — provider scores\n"
                 "(hatched=B, solid=A; blue=P2 nom., green=P3 adj.)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    _style(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "provider_scores")

    # ── Plot 6: Summary — mean utility per policy ─────────────────────────────
    mean_utils = {}
    for pol in POLICIES:
        all_u = [v for vals in util_by[pol].values() for v in vals]
        mean_utils[pol] = float(np.mean(all_u))

    fig, ax = plt.subplots(figsize=(COL_W, ROW_H * 1.2))
    colors  = [POLICY_STYLE[p][0] for p in POLICIES]
    names   = [POLICY_STYLE[p][3] for p in POLICIES]
    vals    = [mean_utils[p] for p in POLICIES]
    bars = ax.bar(names, vals, color=colors, alpha=0.82, edgecolor="white", linewidth=0.8,
                  width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, fontweight="bold")
    ax.set_ylabel("Mean true utility")
    ax.set_title(f"{label} — mean utility per policy\n"
                 f"({n_runs} runs × {sim_t + 1} steps)")
    ax.set_ylim(0, max(vals) * 1.18)
    _style(ax)
    fig.tight_layout(pad=0.4)
    _save(fig, fig_dir, "summary")

    if util_a_by and util_b_by:
        _plot_provider_comparison(data_dir, fig_dir, scenario_key, cfg,
                                  util_by, util_a_by, util_b_by, ts)

    print(f"  [{scenario_key}] all plots done", flush=True)


def _nominal_utility(nom: dict, w_acc: float, w_lat: float, w_rel: float) -> float:
    return compute_utility(nom["acc_mean"], nom["lat_mean"], nom["rel_mean"],
                           w_acc, w_lat, w_rel)


def _risk_adj_utility(nom: dict, risk_q: float, risk_a: float,
                      w_acc: float, w_lat: float, w_rel: float,
                      sigma_q: float, sigma_a: float) -> float:
    return compute_risk_adjusted_utility(
        nom["acc_mean"], nom["lat_mean"], nom["rel_mean"],
        sigma_q, sigma_a, risk_q, risk_a, w_acc, w_lat, w_rel)


def _legend(ax, policies: list) -> None:
    handles = []
    for pol in policies:
        c, ls, mk, name = POLICY_STYLE[pol]
        handles.append(mlines.Line2D([], [], color=c, linewidth=LW, linestyle=ls,
                                     marker=mk, markersize=MS, label=f"{name} (med.)"))
        handles.append(mlines.Line2D([], [], color=c, linewidth=LW_BAND,
                                     linestyle="--", alpha=0.70,
                                     label=f"{name} (p10/p90)"))
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="lower left",
              framealpha=0.92, ncol=1)


# ── File helpers ──────────────────────────────────────────────────────────────

def _write_csv(path: Path, header: list, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run uncertainty-aware selection simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario", choices=["1", "2", "3", "all"], default="all",
                   help="Which scenario preset to run")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated scenario keys, e.g. scenario_2_idt,scenario_1")
    p.add_argument("--runs",     type=int, default=30,
                   help="Number of randomised simulation runs")
    p.add_argument("--seed",     type=int, default=None,
                   help="Base seed (auto-generated if omitted)")
    p.add_argument("--output-dir", type=Path, default=Path("results/uncertainty_simulation"),
                   help="Root directory for CSV output")
    p.add_argument("--figures-dir", type=Path, default=Path("docs/figures/uncertainty_simulation"),
                   help="Root directory for figure output")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip simulation, only regenerate plots from existing CSV")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    if args.scenarios is not None:
        scenario_keys = [k.strip() for k in args.scenarios.split(",")]
    elif args.scenario == "all":
        scenario_keys = ["scenario_1", "scenario_2", "scenario_3"]
    else:
        scenario_keys = [f"scenario_{args.scenario}"]

    print(f"\n{'='*62}")
    print(f"  Base seed    : {base_seed}")
    print(f"  Runs         : {args.runs}")
    print(f"  Scenarios    : {', '.join(scenario_keys)}")
    print(f"  Output dir   : {args.output_dir.resolve()}")
    print(f"  Figures dir  : {args.figures_dir.resolve()}")
    print(f"  Plot only    : {args.plot_only}")
    print(f"{'='*62}\n")

    for sc_key in scenario_keys:
        cfg        = UNCERTAINTY_SCENARIOS[sc_key]
        out_dir    = args.output_dir  / sc_key
        fig_dir    = args.figures_dir / sc_key

        print(f"[{sc_key}] {cfg['label']}")

        if not args.plot_only:
            run_scenario(sc_key, args.runs, base_seed, out_dir, cfg)
            print()

        print(f"[{sc_key}] Generating plots…")
        _generate_plots(out_dir, fig_dir, sc_key, cfg)
        print()

    print(f"{'='*62}")
    print(f"  Done.  Base seed: {base_seed}")
    print(f"  Reproduce: python scripts/uncertainty_sim.py "
          f"--seed {base_seed} --runs {args.runs} --scenario {args.scenario}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
