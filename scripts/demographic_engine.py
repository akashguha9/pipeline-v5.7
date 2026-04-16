"""
demographic_engine.py
======================
Layer 0 of S7 REGIME — Demographic Gravity Field.
Computes birth gravity B_i, velocity dB_i/dt, age pressure,
and structural demand flow signals for the pipeline.

Pipeline position: feeds S7 regime detection + S3B capital flow
Integration: run daily/weekly, output feeds timeseries dataset as new columns

Data: static CSV or UN World Population Prospects API
Cost: $0/month (static) to ~$0/month (free UN data API)
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


# ── STATIC DEMOGRAPHIC DATA (2024 estimates) ──────────────────────────────────
# Source: UN World Population Prospects 2024, World Bank
# Update annually or pull from UN API

DEMOGRAPHIC_DATA = {
    # region: {pop_millions, fertility_rate, median_age, urbanization_pct}
    "Africa":           {"pop": 1470, "fertility": 4.2, "median_age": 19.7, "urban": 44},
    "South_Asia":       {"pop": 1980, "fertility": 2.1, "median_age": 28.1, "urban": 38},
    "Southeast_Asia":   {"pop":  690, "fertility": 2.0, "median_age": 30.2, "urban": 51},
    "East_Asia":        {"pop": 1650, "fertility": 1.2, "median_age": 39.8, "urban": 66},
    "Europe":           {"pop":  750, "fertility": 1.5, "median_age": 43.9, "urban": 75},
    "North_America":    {"pop":  370, "fertility": 1.7, "median_age": 38.6, "urban": 83},
    "Latin_America":    {"pop":  660, "fertility": 1.9, "median_age": 31.0, "urban": 81},
    "Middle_East":      {"pop":  420, "fertility": 2.8, "median_age": 27.5, "urban": 67},
    "Central_Asia":     {"pop":  110, "fertility": 2.6, "median_age": 29.0, "urban": 52},
    "Oceania":          {"pop":   45, "fertility": 2.2, "median_age": 33.0, "urban": 72},
    # Key country-level data
    "India":            {"pop": 1430, "fertility": 2.0, "median_age": 28.4, "urban": 36},
    "China":            {"pop": 1400, "fertility": 1.1, "median_age": 38.9, "urban": 64},
    "USA":              {"pop":  340, "fertility": 1.7, "median_age": 38.3, "urban": 83},
    "Germany":          {"pop":   84, "fertility": 1.5, "median_age": 44.6, "urban": 77},
    "Nigeria":          {"pop":  230, "fertility": 5.1, "median_age": 17.9, "urban": 54},
    "Indonesia":        {"pop":  280, "fertility": 2.1, "median_age": 30.2, "urban": 58},
    "Brazil":           {"pop":  217, "fertility": 1.7, "median_age": 33.5, "urban": 87},
    "Japan":            {"pop":  124, "fertility": 1.2, "median_age": 48.9, "urban": 92},
    "Pakistan":         {"pop":  240, "fertility": 3.4, "median_age": 22.1, "urban": 37},
    "Bangladesh":       {"pop":  170, "fertility": 2.0, "median_age": 28.0, "urban": 40},
}

# Historical fertility data for velocity computation (5-year snapshots)
FERTILITY_HISTORY = {
    "Africa":        {2010: 4.8, 2015: 4.5, 2020: 4.3, 2024: 4.2},
    "Europe":        {2010: 1.6, 2015: 1.58, 2020: 1.52, 2024: 1.5},
    "South_Asia":    {2010: 2.6, 2015: 2.4, 2020: 2.2, 2024: 2.1},
    "East_Asia":     {2010: 1.5, 2015: 1.4, 2020: 1.25, 2024: 1.2},
    "Latin_America": {2010: 2.2, 2015: 2.1, 2020: 1.95, 2024: 1.9},
    "Middle_East":   {2010: 3.2, 2015: 3.0, 2020: 2.9, 2024: 2.8},
    "North_America": {2010: 1.9, 2015: 1.84, 2020: 1.75, 2024: 1.7},
}


# ── CORE COMPUTATIONS ─────────────────────────────────────────────────────────

def compute_birth_gravity(data: dict = DEMOGRAPHIC_DATA) -> pd.DataFrame:
    """
    B_i = (Pop_i × Fertility_i) / Σ_j (Pop_j × Fertility_j)
    Returns birth gravity shares for all regions.
    """
    rows = []
    for region, d in data.items():
        rows.append({
            "region": region,
            "pop": d["pop"],
            "fertility": d["fertility"],
            "median_age": d["median_age"],
            "urban": d["urban"],
            "raw_births": d["pop"] * d["fertility"],
        })
    df = pd.DataFrame(rows)
    total_births = df["raw_births"].sum()
    df["birth_gravity"] = df["raw_births"] / total_births
    df["birth_gravity_pct"] = (df["birth_gravity"] * 100).round(2)
    return df.sort_values("birth_gravity", ascending=False).reset_index(drop=True)


def compute_fertility_velocity(
    history: dict = FERTILITY_HISTORY,
    base_year: int = 2020,
    current_year: int = 2024,
) -> pd.DataFrame:
    """
    dB_i/dt ≈ (Fertility_{t} - Fertility_{t-k}) / k
    Returns directional velocity of fertility rates.
    Positive = rising fertility (young population growing)
    Negative = falling fertility (aging population trend)
    """
    rows = []
    years_elapsed = current_year - base_year
    for region, hist in history.items():
        if base_year in hist and current_year in hist:
            f_base = hist[base_year]
            f_now  = hist[current_year]
            velocity = (f_now - f_base) / years_elapsed
            pct_change = (f_now - f_base) / f_base * 100
            rows.append({
                "region": region,
                "fertility_base": f_base,
                "fertility_now": f_now,
                "velocity": round(velocity, 4),
                "pct_change": round(pct_change, 2),
                "direction": "RISING" if velocity > 0 else "FALLING",
            })
    return pd.DataFrame(rows).sort_values("velocity", ascending=False)


def compute_age_pressure(data: dict = DEMOGRAPHIC_DATA) -> pd.DataFrame:
    """
    Compute aging pressure and structural demand signals.
    Median age > 40 → elder-care, healthcare, automation demand
    Median age < 25 → education, telecom, payments, FMCG demand
    """
    rows = []
    for region, d in data.items():
        age = d["median_age"]
        urb = d["urban"]
        fert = d["fertility"]

        # Age pressure score: higher = older population
        age_pressure = (age - 20) / 40  # normalized 0-1 for age 20-60

        # Youth bulge score: higher = more youth demand
        youth_bulge = max(0, (3.5 - fert) / 3.5 * -1 + 1)  # normalized

        # Urban frontier score: potential for urban infrastructure build
        urban_frontier = 1 - (urb / 100)

        # Structural demand classification
        if age < 25:
            demand_type = "YOUTH_GROWTH"
            sectors = "education,telecom,payments,FMCG,healthcare,energy"
        elif age < 35:
            demand_type = "WORKING_AGE"
            sectors = "housing,consumer,productivity,financial_services"
        elif age < 42:
            demand_type = "TRANSITION"
            sectors = "healthcare,savings,insurance,automation"
        else:
            demand_type = "AGING"
            sectors = "elder-care,healthcare,pensions,automation,migration_politics"

        rows.append({
            "region": region,
            "median_age": age,
            "age_pressure": round(age_pressure, 3),
            "youth_bulge": round(youth_bulge, 3),
            "urban_frontier": round(urban_frontier, 3),
            "demand_type": demand_type,
            "structural_sectors": sectors,
        })
    return pd.DataFrame(rows).sort_values("age_pressure", ascending=False)


def compute_migration_pressure(data: dict = DEMOGRAPHIC_DATA) -> pd.DataFrame:
    """
    Migration pressure = differential between high-fertility young regions
    and low-fertility aging regions.
    High dP → migration flow potential → political narrative signal
    """
    df = compute_birth_gravity(data)
    young = df[df["median_age"] < 30]["birth_gravity"].sum()
    old   = df[df["median_age"] > 40]["birth_gravity"].sum()
    differential = young - old

    rows = []
    for _, row in df.iterrows():
        if row["median_age"] < 30 and row["fertility"] > 2.5:
            direction = "SENDER"  # likely migration source
            pressure = row["birth_gravity"] * (row["fertility"] - 2.5) / 2.5
        elif row["median_age"] > 38 and row["fertility"] < 1.8:
            direction = "RECEIVER"  # likely migration destination
            pressure = row["birth_gravity"] * (1.8 - row["fertility"]) / 1.8
        else:
            direction = "NEUTRAL"
            pressure = 0.0
        rows.append({
            "region": row["region"],
            "median_age": row["median_age"],
            "fertility": row["fertility"],
            "migration_role": direction,
            "migration_pressure": round(float(pressure), 4),
        })
    return pd.DataFrame(rows).sort_values("migration_pressure", ascending=False)


# ── SECOND-ORDER CHAIN GENERATOR ──────────────────────────────────────────────

def generate_second_order_chains(data: dict = DEMOGRAPHIC_DATA) -> list[dict]:
    """
    Generates structural second-order signal chains.
    Chain: Demographic_condition → Layer2 → Layer3 → Market_outcome
    """
    df_age = compute_age_pressure(data)
    df_mig = compute_migration_pressure(data)
    chains = []

    # Chain 1: Birth concentration → labor growth → migration → politics
    high_birth = df_mig[df_mig["migration_role"] == "SENDER"]["region"].tolist()
    if high_birth:
        chains.append({
            "chain_id": "BIRTH_MIGRATION_POLITICS",
            "trigger": f"Birth concentration: {', '.join(high_birth[:3])}",
            "layer_1": "Labor force growth → surplus labor supply",
            "layer_2": "Economic migration pressure → flows to aging regions",
            "layer_3": "Host country political response → immigration policy",
            "market_outcome": "European politics sensitivity · border security · remittance flows",
            "kalshi_proxy": "European election outcomes · refugee policy markets",
            "horizon": "MEDIUM (1-5 years)",
            "signal_strength": "STRUCTURAL",
        })

    # Chain 2: Aging population → healthcare demand → policy
    aging = df_age[df_age["demand_type"] == "AGING"]["region"].tolist()
    if aging:
        chains.append({
            "chain_id": "AGING_HEALTHCARE_POLICY",
            "trigger": f"Aging: {', '.join(aging[:3])}",
            "layer_1": "Population aging → rising chronic disease burden",
            "layer_2": "Healthcare demand growth → system cost pressure",
            "layer_3": "Policy response → tax/spending shifts",
            "market_outcome": "Healthcare equities · elder-care · automation demand",
            "kalshi_proxy": "Healthcare policy markets · pension reform odds",
            "horizon": "LONG (5-20 years)",
            "signal_strength": "PERSISTENT_STRUCTURAL",
        })

    # Chain 3: Urban frontier → infrastructure build
    urban_frontier = df_age[df_age["urban_frontier"] > 0.5]["region"].tolist()
    if urban_frontier:
        chains.append({
            "chain_id": "URBANIZATION_INFRASTRUCTURE",
            "trigger": f"Urban frontier: {', '.join(urban_frontier[:3])}",
            "layer_1": "Low urbanization → large urban migration flows",
            "layer_2": "Infrastructure demand → housing/telecom/payments build",
            "layer_3": "Consumer class formation → FMCG/financial services",
            "market_outcome": "Emerging market consumer · telecom · payment companies",
            "kalshi_proxy": "Emerging market economic growth · IPO activity",
            "horizon": "MEDIUM-LONG",
            "signal_strength": "STRUCTURAL",
        })

    return chains


# ── PIPELINE OUTPUT ────────────────────────────────────────────────────────────

def run_demographic_engine(
    output_dir: str = ".",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Master runner. Produces all demographic signal tables.
    Returns dict of DataFrames for pipeline integration.
    """
    results = {}

    # Core computations
    results["birth_gravity"]    = compute_birth_gravity()
    results["fertility_velocity"] = compute_fertility_velocity()
    results["age_pressure"]     = compute_age_pressure()
    results["migration_pressure"] = compute_migration_pressure()

    # Second-order chains
    chains = generate_second_order_chains()
    results["second_order_chains"] = pd.DataFrame(chains)

    # Top structural signal: highest birth gravity + rising velocity
    bg = results["birth_gravity"].set_index("region")
    fv = results["fertility_velocity"].set_index("region")
    age = results["age_pressure"].set_index("region")[["age_pressure","demand_type","structural_sectors"]]

    # Unified demographic signal score
    # Score = birth_gravity_weight × (1 + velocity_boost) × urban_frontier
    signal_rows = []
    for region in bg.index:
        b_grav = bg.loc[region, "birth_gravity"] if region in bg.index else 0
        vel    = fv.loc[region, "velocity"] if region in fv.index else 0
        a_pres = age.loc[region, "age_pressure"] if region in age.index else 0.5
        sectors = age.loc[region, "structural_sectors"] if region in age.index else "N/A"

        # Demographic alpha score: high birth gravity + falling fertility (transition signal)
        # OR high age pressure (aging structural signal)
        demo_alpha = b_grav * 0.4 + abs(vel) * 10 * 0.3 + a_pres * 0.3
        signal_rows.append({
            "region": region,
            "birth_gravity": round(b_grav, 4),
            "fertility_velocity": round(float(vel), 4),
            "age_pressure": round(float(a_pres), 3),
            "demo_alpha_score": round(demo_alpha, 4),
            "structural_sectors": sectors,
        })
    results["demographic_alpha"] = pd.DataFrame(signal_rows).sort_values(
        "demo_alpha_score", ascending=False
    )

    if verbose:
        print(f"\n{'='*60}")
        print("DEMOGRAPHIC ENGINE OUTPUT")
        print(f"{'='*60}")
        print("\nTop Birth Gravity Regions:")
        print(results["birth_gravity"][["region","birth_gravity_pct","fertility","median_age"]].head(5).to_string(index=False))
        print("\nFertility Velocity (biggest movers):")
        print(results["fertility_velocity"][["region","velocity","direction"]].head(5).to_string(index=False))
        print("\nDemographic Alpha Scores:")
        print(results["demographic_alpha"][["region","demo_alpha_score","structural_sectors"]].head(8).to_string(index=False))
        print(f"\nSecond-order chains generated: {len(chains)}")
        for c in chains:
            outcome = c.get('market_outcome', c.get('market_output', 'N/A'))
            print(f"  → {c['chain_id']}: {str(outcome)[:60]}")

    # Save outputs
    ts = datetime.now().strftime("%Y%m%d")
    for name, df in results.items():
        path = os.path.join(output_dir, f"demographic_{name}_{ts}.csv")
        df.to_csv(path, index=False)

    return results


# ── INTEGRATION WITH signal_engine.py ─────────────────────────────────────────

def get_demographic_context_for_seed(
    seed_label: str,
    results: Optional[dict] = None,
) -> dict:
    """
    Returns demographic context for a given prediction market seed label.
    Maps seed keywords to relevant demographic signals.
    Used to add structural backdrop to signal quality scoring.
    """
    if results is None:
        results = run_demographic_engine(verbose=False)

    da = results["demographic_alpha"].set_index("region")
    chains = results["second_order_chains"] if "second_order_chains" in results else pd.DataFrame()

    # Keyword → demographic region mapping
    region_keywords = {
        "africa": "Africa", "nigeria": "Nigeria", "india": "India",
        "china": "China", "europe": "Europe", "germany": "Germany",
        "japan": "Japan", "brazil": "Brazil", "usa": "USA",
        "migration": "Africa",  # migration narratives → demographic root
        "healthcare": "Europe",  # aging Europe → healthcare
        "telecom": "Africa",    # youth Africa → telecom build
        "aging": "Europe",      "education": "South_Asia",
        "payments": "Africa",   "remittance": "South_Asia",
    }

    matched_region = None
    for kw, region in region_keywords.items():
        if kw in seed_label.lower():
            matched_region = region
            break

    if matched_region and matched_region in da.index:
        row = da.loc[matched_region]
        return {
            "demographic_context": matched_region,
            "birth_gravity": float(row["birth_gravity"]),
            "age_pressure": float(row["age_pressure"]),
            "demo_alpha_score": float(row["demo_alpha_score"]),
            "structural_sectors": row["structural_sectors"],
            "demographic_signal_quality_boost": min(0.15, float(row["demo_alpha_score"])),
        }

    return {
        "demographic_context": "unknown",
        "birth_gravity": 0.0,
        "age_pressure": 0.5,
        "demo_alpha_score": 0.0,
        "structural_sectors": "N/A",
        "demographic_signal_quality_boost": 0.0,
    }


if __name__ == "__main__":
    print("Running Demographic Engine self-test...\n")
    results = run_demographic_engine(output_dir=".", verbose=True)

    print("\n\nDemographic context for sample seeds:")
    for seed in ["africa_telecom", "europe_healthcare", "india_payments", "migration_politics"]:
        ctx = get_demographic_context_for_seed(seed, results)
        print(f"  {seed}: alpha={ctx['demo_alpha_score']:.3f} | {ctx['structural_sectors'][:40]}")

    print("\n✓ Demographic engine self-test complete.")
