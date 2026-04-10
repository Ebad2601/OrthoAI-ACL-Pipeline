"""
OrthoAI ACL Injury Risk — Full Analysis Pipeline
=================================================
A complete end-to-end machine learning pipeline for ACL injury prediction
in competitive athletes. Covers data loading, cleaning, exploratory data
analysis (EDA), feature engineering, model training, and validation.

Steps
-----
1. Load & Audit Data
2. Exploratory Data Analysis  → outputs/eda_*.png
3. Feature Engineering
4. Model Training  (Logistic Regression, Random Forest, Gradient Boosting)
5. Evaluation  (AUC-ROC, Precision-Recall, Feature Importance)
6. Save best model  → models/best_model.pkl

Author : OrthoAI Research — Muhammad Ebadur Rahman Siddiqui
Dataset: Synthetic ACL Biomechanical Dataset v1.0 (N=1500 athlete-seasons)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import seaborn             as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline          import Pipeline
from sklearn.inspection        import permutation_importance

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data",    "acl_injury_dataset.csv")
OUTDIR = os.path.join(BASE, "outputs")
MODDIR = os.path.join(BASE, "models")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODDIR, exist_ok=True)

# ── Aesthetic config ──────────────────────────────────────────────────────────
PALETTE  = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#f5a623"]
ACCENT   = "#e94560"
DARK     = "#1a1a2e"
LIGHT_BG = "#f7f9fc"

plt.rcParams.update({
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    LIGHT_BG,
    "axes.edgecolor":    "#cccccc",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# =============================================================================
# STEP 1 — LOAD & AUDIT
# =============================================================================
print("\n" + "="*60)
print("  STEP 1 · LOAD & AUDIT")
print("="*60)

df = pd.read_csv(DATA)
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Target distribution:\n{df['acl_injury_this_season'].value_counts().to_string()}")
print(f"\n  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("  STEP 2 · EXPLORATORY DATA ANALYSIS")
print("="*60)

TARGET = "acl_injury_this_season"
NUMERIC_FEATURES = [
    "age", "bmi", "weekly_training_hours", "acwr", "session_rpe",
    "monotony_index", "knee_valgus_deg", "hop_symmetry_pct", "hq_ratio",
    "landing_force_bw", "sleep_hours", "wellness_score"
]
BINARY_FEATURES  = ["prev_knee_injury", "prev_acl_injury"]
CAT_FEATURES     = ["sex", "sport"]

injured     = df[df[TARGET] == 1]
not_injured = df[df[TARGET] == 0]

# ── Figure 1: Injury overview dashboard ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("OrthoAI ACL Dataset — Overview Dashboard", fontsize=16,
             fontweight="bold", color=DARK, y=1.01)

# 1a — Injury prevalence pie
axes[0,0].pie(
    [len(not_injured), len(injured)],
    labels=["No Injury", "ACL Injury"],
    colors=["#aed6f1", ACCENT],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 11}
)
axes[0,0].set_title("Season Injury Prevalence")

# 1b — Sport breakdown
sport_inj = df.groupby("sport")[TARGET].mean().sort_values(ascending=False)
sport_inj.plot(kind="bar", ax=axes[0,1], color=ACCENT, edgecolor="white", width=0.65)
axes[0,1].set_title("Injury Rate by Sport")
axes[0,1].set_ylabel("Injury Rate")
axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=30, ha="right")

# 1c — Sex breakdown
sex_inj = df.groupby("sex")[TARGET].mean()
sex_inj.plot(kind="bar", ax=axes[0,2], color=["#5dade2", ACCENT], edgecolor="white", width=0.5)
axes[0,2].set_title("Injury Rate by Sex")
axes[0,2].set_ylabel("Injury Rate")
axes[0,2].set_xticklabels(axes[0,2].get_xticklabels(), rotation=0)

# 1d — ACWR distribution
axes[1,0].hist(not_injured["acwr"], bins=35, alpha=0.65, color="#5dade2", label="No Injury")
axes[1,0].hist(injured["acwr"],     bins=35, alpha=0.75, color=ACCENT,    label="ACL Injury")
axes[1,0].axvline(1.5, color=DARK, linestyle="--", linewidth=1.2, label="Danger Zone (1.5)")
axes[1,0].set_title("ACWR Distribution by Injury Status")
axes[1,0].set_xlabel("Acute:Chronic Workload Ratio")
axes[1,0].legend(fontsize=8)

# 1e — Knee Valgus
axes[1,1].hist(not_injured["knee_valgus_deg"], bins=35, alpha=0.65, color="#5dade2", label="No Injury")
axes[1,1].hist(injured["knee_valgus_deg"],     bins=35, alpha=0.75, color=ACCENT,    label="ACL Injury")
axes[1,1].set_title("Knee Valgus Angle Distribution")
axes[1,1].set_xlabel("Valgus Angle (degrees)")
axes[1,1].legend(fontsize=8)

# 1f — H:Q Ratio
axes[1,2].hist(not_injured["hq_ratio"], bins=35, alpha=0.65, color="#5dade2", label="No Injury")
axes[1,2].hist(injured["hq_ratio"],     bins=35, alpha=0.75, color=ACCENT,    label="ACL Injury")
axes[1,2].axvline(0.60, color=DARK, linestyle="--", linewidth=1.2, label="Risk Threshold (<0.60)")
axes[1,2].set_title("Hamstring:Quadriceps Ratio")
axes[1,2].set_xlabel("H:Q Ratio")
axes[1,2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "eda_1_overview_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: eda_1_overview_dashboard.png")

# ── Figure 2: Correlation heatmap ─────────────────────────────────────────────
corr_df = df[NUMERIC_FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
    center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Matrix — OrthoAI ACL Dataset", fontsize=14, pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "eda_2_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: eda_2_correlation_heatmap.png")

# ── Figure 3: Boxplots — injured vs not ──────────────────────────────────────
key_features = ["acwr", "knee_valgus_deg", "hq_ratio", "hop_symmetry_pct",
                "sleep_hours", "landing_force_bw"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    data_0 = df[df[TARGET]==0][feat].dropna()
    data_1 = df[df[TARGET]==1][feat].dropna()
    bp = axes[i].boxplot(
        [data_0, data_1],
        labels=["No Injury", "ACL Injury"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2}
    )
    bp["boxes"][0].set_facecolor("#aed6f1")
    bp["boxes"][1].set_facecolor(ACCENT)
    axes[i].set_title(feat.replace("_", " ").title())
fig.suptitle("Key Risk Factors: Injured vs Non-Injured Athletes", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "eda_3_boxplots_risk_factors.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: eda_3_boxplots_risk_factors.png")

# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60)
print("  STEP 3 · FEATURE ENGINEERING")
print("="*60)

df_ml = df.copy()

# Encode categoricals
le_sex   = LabelEncoder()
le_sport = LabelEncoder()
df_ml["sex_enc"]   = le_sex.fit_transform(df_ml["sex"])
df_ml["sport_enc"] = le_sport.fit_transform(df_ml["sport"])

# Composite risk flags (clinically grounded)
df_ml["high_acwr"]          = (df_ml["acwr"]           > 1.5).astype(int)
df_ml["poor_hq_ratio"]      = (df_ml["hq_ratio"]       < 0.60).astype(int)
df_ml["limb_asymmetry"]     = (df_ml["hop_symmetry_pct"] < 90).astype(int)
df_ml["high_valgus"]        = (df_ml["knee_valgus_deg"] > 8).astype(int)
df_ml["sleep_deprived"]     = (df_ml["sleep_hours"]    < 6).astype(int)
df_ml["composite_risk"]     = (
    df_ml["high_acwr"] + df_ml["poor_hq_ratio"] +
    df_ml["limb_asymmetry"] + df_ml["high_valgus"]
)

FEATURE_COLS = (
    NUMERIC_FEATURES + BINARY_FEATURES +
    ["sex_enc", "sport_enc", "high_acwr", "poor_hq_ratio",
     "limb_asymmetry", "high_valgus", "sleep_deprived", "composite_risk"]
)

X = df_ml[FEATURE_COLS]
y = df_ml[TARGET]

print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Class balance: {y.value_counts().to_dict()}")

# Train / test split — stratified (preserves injury ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# =============================================================================
# STEP 4 — MODEL TRAINING
# =============================================================================
print("\n" + "="*60)
print("  STEP 4 · MODEL TRAINING")
print("="*60)

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

models = {
    "Logistic Regression": Pipeline([
        ("impute",  SimpleImputer(strategy="median")),
        ("scale",   StandardScaler()),
        ("clf",     LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("impute",  SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("impute",  SimpleImputer(strategy="median")),
        ("clf",     GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ))
    ]),
}

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    results[name] = {"cv_auc": cv_auc.mean(), "cv_std": cv_auc.std(), "test_auc": test_auc, "pipe": pipe}
    print(f"  {name:25s}  CV AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}  |  Test AUC={test_auc:.3f}")

best_name = max(results, key=lambda k: results[k]["test_auc"])
best_pipe = results[best_name]["pipe"]
print(f"\n  ★ Best model: {best_name}  (Test AUC={results[best_name]['test_auc']:.3f})")

# =============================================================================
# STEP 5 — EVALUATION
# =============================================================================
print("\n" + "="*60)
print("  STEP 5 · EVALUATION")
print("="*60)

y_prob = best_pipe.predict_proba(X_test)[:,1]
y_pred = best_pipe.predict(X_test)

print("\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Injury", "ACL Injury"]))

# ── Figure 4: Model Evaluation ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# 4a — ROC curves for all models
ax_roc = fig.add_subplot(gs[0, 0])
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["pipe"].predict_proba(X_test)[:,1])
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['test_auc']:.3f})")
ax_roc.plot([0,1],[0,1],"--", color="#999", linewidth=1)
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — All Models"); ax_roc.legend(fontsize=8)

# 4b — Precision-Recall curve (best model)
ax_pr = fig.add_subplot(gs[0, 1])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
ax_pr.plot(rec, prec, color=ACCENT, linewidth=2, label=f"AP={ap:.3f}")
ax_pr.axhline(y_test.mean(), color="#999", linestyle="--", label="Baseline")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"Precision-Recall — {best_name}"); ax_pr.legend(fontsize=9)

# 4c — Confusion matrix
ax_cm = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Injury", "ACL Injury"])
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(f"Confusion Matrix — {best_name}")

# 4d — Cross-validation AUC comparison
ax_cv = fig.add_subplot(gs[1, 0])
names_  = list(results.keys())
aucs_   = [results[n]["cv_auc"] for n in names_]
stds_   = [results[n]["cv_std"] for n in names_]
colors_ = [ACCENT if n==best_name else "#5dade2" for n in names_]
bars = ax_cv.bar(names_, aucs_, yerr=stds_, capsize=5, color=colors_, edgecolor="white", width=0.5)
ax_cv.set_ylim(0, 1); ax_cv.set_ylabel("AUC-ROC")
ax_cv.set_title("5-Fold Cross-Validation AUC")
ax_cv.set_xticklabels(names_, rotation=15, ha="right")

# 4e — Feature importance (best tree model or permutation)
ax_fi = fig.add_subplot(gs[1, 1:])
if hasattr(best_pipe["clf"], "feature_importances_"):
    importances = best_pipe["clf"].feature_importances_
else:
    r = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42)
    importances = r.importances_mean

fi_df = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
fi_df.plot(kind="barh", ax=ax_fi, color=ACCENT, edgecolor="white")
ax_fi.set_title(f"Feature Importance — {best_name} (Top 15)")
ax_fi.set_xlabel("Importance Score")

fig.suptitle(f"OrthoAI ACL Prediction — Model Evaluation Dashboard\n"
             f"Best: {best_name} | Test AUC = {results[best_name]['test_auc']:.3f}",
             fontsize=14, fontweight="bold", color=DARK)
fig.savefig(os.path.join(OUTDIR, "evaluation_model_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: evaluation_model_dashboard.png")

# =============================================================================
# STEP 6 — SAVE MODEL
# =============================================================================
print("\n" + "="*60)
print("  STEP 6 · SAVE MODEL")
print("="*60)

model_path = os.path.join(MODDIR, "best_model.pkl")
joblib.dump({"model": best_pipe, "features": FEATURE_COLS, "name": best_name}, model_path)
print(f"  ✓ Model saved: {model_path}")

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
print(f"""
  Summary
  -------
  Dataset     : 1,500 athlete-seasons, 19 features
  Best Model  : {best_name}
  Test AUC    : {results[best_name]['test_auc']:.3f}
  CV AUC      : {results[best_name]['cv_auc']:.3f} ± {results[best_name]['cv_std']:.3f}
  Outputs     : outputs/ (4 publication-quality figures)
  Model file  : models/best_model.pkl
""")
