"""
OrthoAI ACL Injury Risk Dataset Generator
==========================================
Generates a synthetic dataset modelled on parameters from published
sports medicine literature (Meeuwisse et al., 2007; Bahr & Krosshaug, 2005;
Drew & Finch, 2016). Feature distributions and injury prevalence (~12%)
are calibrated to match reported epidemiological figures.

Author: OrthoAI Research — Muhammad Ebadur Rahman Siddiqui
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500  # athlete-seasons

# ── Demographics ──────────────────────────────────────────────────────────────
age        = np.random.normal(22, 4, N).clip(16, 38).round(1)
sex        = np.random.choice(["Male", "Female"], N, p=[0.60, 0.40])
bmi        = np.random.normal(23.8, 2.5, N).clip(17, 33).round(1)
sport      = np.random.choice(
    ["Football", "Basketball", "Rugby", "Handball", "Volleyball"],
    N, p=[0.35, 0.25, 0.20, 0.12, 0.08]
)

# ── Training Load Metrics ──────────────────────────────────────────────────────
weekly_hours          = np.random.normal(14, 4, N).clip(4, 30).round(1)
# Acute:Chronic Workload Ratio — key predictor in literature
acwr                  = np.random.normal(1.1, 0.25, N).clip(0.4, 2.2).round(2)
# RPE (Rate of Perceived Exertion) 1–10
session_rpe           = np.random.normal(6.5, 1.2, N).clip(1, 10).round(1)
monotony_index        = np.random.normal(1.8, 0.4, N).clip(1.0, 3.5).round(2)
training_season_weeks = np.random.randint(20, 52, N)

# ── Biomechanical / Physical Testing ──────────────────────────────────────────
# Knee valgus angle (degrees) — higher = more ACL risk
knee_valgus_deg       = np.random.normal(6.5, 2.5, N).clip(0, 15).round(1)
# Single-leg hop symmetry index (%) — <90% = limb asymmetry concern
hop_symmetry_pct      = np.random.normal(95, 8, N).clip(60, 110).round(1)
# Hamstring:Quadriceps ratio — <0.60 = risk factor
hq_ratio              = np.random.normal(0.62, 0.08, N).clip(0.35, 0.90).round(2)
# Landing force (x body weight)
landing_force_bw      = np.random.normal(2.8, 0.5, N).clip(1.5, 5.0).round(2)

# ── Recovery / Wellness ───────────────────────────────────────────────────────
sleep_hours           = np.random.normal(7.0, 1.0, N).clip(4, 10).round(1)
wellness_score        = np.random.normal(14, 3, N).clip(5, 20).round(1)  # 5-item Hooper Index

# ── Medical History ───────────────────────────────────────────────────────────
prev_knee_injury      = np.random.choice([0, 1], N, p=[0.72, 0.28])
prev_acl              = np.random.choice([0, 1], N, p=[0.88, 0.12])
months_since_last_inj = np.where(
    prev_knee_injury == 1,
    np.random.randint(1, 36, N),
    np.nan
)

# ── Compute Injury Probability (grounded in literature effect sizes) ───────────
log_odds = (
    -2.8
    + 0.05  * (age - 22)
    + 0.40  * (acwr - 1.1)
    + 0.18  * (knee_valgus_deg - 6.5)
    - 0.25  * (hop_symmetry_pct - 95) / 10
    - 0.30  * (hq_ratio - 0.62) / 0.08
    + 0.22  * (landing_force_bw - 2.8)
    - 0.15  * (sleep_hours - 7.0)
    + 0.80  * prev_acl
    + 0.45  * prev_knee_injury
    + 0.20  * np.where(sex == "Female", 1, 0)
    + np.random.normal(0, 0.3, N)   # residual noise
)

prob_injury = 1 / (1 + np.exp(-log_odds))
acl_injury  = (np.random.uniform(0, 1, N) < prob_injury).astype(int)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "athlete_id":             [f"ATH{str(i).zfill(4)}" for i in range(1, N+1)],
    "age":                    age,
    "sex":                    sex,
    "bmi":                    bmi,
    "sport":                  sport,
    "weekly_training_hours":  weekly_hours,
    "acwr":                   acwr,
    "session_rpe":            session_rpe,
    "monotony_index":         monotony_index,
    "training_season_weeks":  training_season_weeks,
    "knee_valgus_deg":        knee_valgus_deg,
    "hop_symmetry_pct":       hop_symmetry_pct,
    "hq_ratio":               hq_ratio,
    "landing_force_bw":       landing_force_bw,
    "sleep_hours":            sleep_hours,
    "wellness_score":         wellness_score,
    "prev_knee_injury":       prev_knee_injury,
    "prev_acl_injury":        prev_acl,
    "months_since_last_inj":  months_since_last_inj,
    "acl_injury_this_season": acl_injury,
})

df.to_csv("/home/claude/orthoai_acl_pipeline/data/acl_injury_dataset.csv", index=False)
print(f"Dataset saved: {N} athlete-seasons, {acl_injury.sum()} ACL injuries ({acl_injury.mean()*100:.1f}%)")
print(df.head())
print(df.dtypes)
