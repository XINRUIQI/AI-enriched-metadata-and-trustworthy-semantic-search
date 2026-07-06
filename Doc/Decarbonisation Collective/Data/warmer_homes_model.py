#!/usr/bin/env python3
"""
================================================================================
WARMER HOMES FOR LONDON  --  Retrofit Prioritisation Engine  (MVP v1)
================================================================================
A borough-agnostic, *interpretable* pipeline built on the London Building Stock
Model v2 (LBSMv2). Point it at any borough's LBSMv2 CSV and it will:

    1. Clean & gate the data (residential, reliable rows only).
    2. Run a LAYERED DECISION FUNNEL that categorises every property by:
         NEED    -> is it already at EPC C (the London target)?
         BENEFIT -> can standard measures get it to C? (interpretable ML model)
         EQUITY  -> can the household self-fund, or does it need public support?
         ROUTE   -> cheapest high-impact measure, constraints, who owns the action.
    3. Train a shallow, fully explainable Decision Tree that predicts whether a
       property is a "retrofit-to-C candidate" from its PHYSICAL fabric alone,
       and exports the human-readable rules + feature importances + metrics.
    4. Export everything a resident / councillor / GLA / landlord would need:
         - property_scores.csv    (one row per property: tier, segment, action, owner)
         - lsoa_summary.csv       (neighbourhood roll-up for targeting)
         - decision_tree.png      (the model, as a picture)
         - decision_tree_rules.txt(the model, as IF/THEN rules)
         - feature_importance.csv (what physically drives priority)
         - model_metrics.json     (accuracy / precision / recall / ROC-AUC ...)

USAGE
    python warmer_homes_model.py --input LBSMv2_Hammersmith_Fulham.csv \
                                 --outdir ./out

Everything here is deliberately transparent: the ML model is one shallow tree,
and the priority score is a documented weighted blend, not a black box. That is
a design choice -- councils and residents have to be able to trust and audit it.
================================================================================
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIG  -- everything policy-relevant lives here so it can be tuned & audited.
# ------------------------------------------------------------------------------

# EPC rating bands, ordered worst -> best. London's target is EPC C by 2030.
EPC_ORDER = ["F-G", "E", "D", "C", "A-B"]
TARGET_RATING = "C"                       # the line every home should get over
AT_OR_ABOVE_TARGET = {"C", "A-B"}         # ratings that already meet the target

# Physical + contextual features the ML model is ALLOWED to see.
# NOTE: we deliberately EXCLUDE current/potential EPC score & rating -- they
# define the label, so letting the model see them would be target leakage.
CATEGORICAL_FEATURES = [
    "property_type", "built_form", "construction_age_band", "tenure",
    "number_habitable_rooms", "wall_type", "wall_insulation",
    "roof_type", "roof_insulation", "glazing_type",
    "main_heat_type", "main_fuel_type",
    "conservation_area_flag", "is_listed",
]
# We also deliberately EXCLUDE energy_consumption from the classifier: current
# annual consumption is essentially a proxy for the CURRENT EPC, so it predicts
# the "needs work" half of the label almost tautologically and swamps the fabric
# levers. We want the tree to explain retrofit POTENTIAL via things you can act
# on, so we keep energy_consumption for context/scoring only, not as a feature.
NUMERIC_FEATURES = [
    "total_floor_area", "estimated_floor_count",
    "solar_pv_potential", "imd19_income_decile", "fuel_poverty",
    "heat_risk_quintile",
]

# Weights for the transparent priority score (Layers 1,3). Documented on purpose.
W_NEED    = 0.35   # how far below EPC C the property is now
W_BENEFIT = 0.30   # how big the achievable EPC uplift is
W_EQUITY  = 0.35   # fuel poverty + income deprivation (who most needs help)

RANDOM_STATE = 42
TREE_MAX_DEPTH = 6          # shallow == readable. This is a feature, not a bug.
TREE_MIN_LEAF  = 200        # each rule must cover a meaningful number of homes


# ------------------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------------------
def load_borough(path):
    """Robustly read an LBSMv2 borough CSV (files ship as latin-1)."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[load] {os.path.basename(path)}: {len(df):,} rows x "
                  f"{df.shape[1]} cols  (encoding={enc})")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with common encodings.")


# ------------------------------------------------------------------------------
# 2. CLEAN  +  GATE  +  FEATURE ENGINEERING
# ------------------------------------------------------------------------------
def clean_and_filter(df):
    """
    The 'data gate' of the funnel. Keep only rows we can stand behind:
      * residential dwellings,
      * every *_known flag == 1 (i.e. no modelled/guessed attributes),
    then handle the handful of awkward columns in the simplest safe way.
    """
    df = df.copy()
    n0 = len(df)

    # -- residential only -------------------------------------------------------
    if "building_use" in df.columns:
        df = df[df["building_use"] == "residential only"]

    # -- keep only fully-known rows (user requirement for the MVP) --------------
    known_cols = [c for c in df.columns if c.endswith("_known")]
    if known_cols:
        mask = (df[known_cols] == 1).all(axis=1)
        df = df[mask]
    # the flags are now all 1 -> constant -> useless as features -> drop them
    df = df.drop(columns=known_cols)

    print(f"[gate] {n0:,} -> {len(df):,} rows "
          f"(residential + all attributes known = {len(df)/n0*100:.1f}%)")

    # -- awkward columns, handled the simplest robust way -----------------------
    # basement_floor: ~90% NaN, where NaN means 'no basement' -> 0
    if "basement_floor" in df:
        df["basement_floor"] = df["basement_floor"].fillna(0)
    # solar fields: NaN means 'no viable roof area' -> 0
    for c in ["solar_pv_area", "solar_pv_potential_11.9", "avg_tilt"]:
        if c in df:
            df[c] = df[c].fillna(0)
    # tidy the awkward solar column name
    if "solar_pv_potential_11.9" in df:
        df = df.rename(columns={"solar_pv_potential_11.9": "solar_pv_potential"})
    # estimated_floor_count: small NaN -> median
    if "estimated_floor_count" in df:
        med = df["estimated_floor_count"].median()
        df["estimated_floor_count"] = df["estimated_floor_count"].fillna(med)
    # listed buildings: collapse {I, II, II*} -> listed, else not listed
    if "listed_building_grade" in df:
        listed = {"I", "II", "II*"}
        df["is_listed"] = np.where(
            df["listed_building_grade"].isin(listed), "listed", "not_listed")
    else:
        df["is_listed"] = "not_listed"

    # -- drop columns we never model on (IDs, geometry, high-cardinality) -------
    # (kept in the frame only if useful for output joins; see id_cols below)
    drop_cols = [
        "os_topo_toid", "easting", "northing", "loac_group",
        "conservation_area_site_id", "listed_building_grade",
        "solar_pv_area", "avg_tilt",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------
# 3. LABEL  (Layer 1 + Layer 2 target)
# ------------------------------------------------------------------------------
def build_label(df):
    """
    The actionable question the ML model answers:
        "Is this a RETROFIT-TO-C CANDIDATE?"
        = currently below EPC C  AND  standard measures can lift it to C.

    We also tag WHY a property is *not* a candidate, which feeds the funnel:
        already_efficient : already at C or better -> monitor only
        deep_retrofit     : below C AND cannot reach C with standard measures
    """
    df = df.copy()
    cur = df["epc_rating"]
    pot = df["potential_epc_rating"]

    below_target = ~cur.isin(AT_OR_ABOVE_TARGET)
    can_reach_C  = pot.isin(AT_OR_ABOVE_TARGET)

    df["is_retrofit_candidate"] = (below_target & can_reach_C).astype(int)

    df["need_state"] = np.select(
        [~below_target, below_target & can_reach_C, below_target & ~can_reach_C],
        ["already_efficient", "retrofit_to_C", "deep_retrofit"],
        default="unknown",
    )

    # magnitude of the achievable win (used later in the priority score)
    df["epc_uplift"] = (df["potential_epc_score"] - df["epc_score"]).clip(lower=0)
    return df


# ------------------------------------------------------------------------------
# 4. TRAIN THE INTERPRETABLE MODEL  (Layer 2 engine)
# ------------------------------------------------------------------------------
def train_model(df, outdir):
    """Shallow Decision Tree: physical fabric -> retrofit-to-C candidate."""
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[cat + num]
    y = df["is_retrofit_candidate"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])
    clf = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH, min_samples_leaf=TREE_MIN_LEAF,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    model = Pipeline([("prep", pre), ("tree", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)

    # ---- metrics (plain sklearn == source of truth) --------------------------
    p_te = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "positive_rate": float(y.mean()),
        "accuracy":  round(accuracy_score(y_te, p_te), 4),
        "precision": round(precision_score(y_te, p_te, zero_division=0), 4),
        "recall":    round(recall_score(y_te, p_te, zero_division=0), 4),
        "f1":        round(f1_score(y_te, p_te, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_te, proba), 4),
        "confusion_matrix": confusion_matrix(y_te, p_te).tolist(),
    }
    print("\n[model] test-set performance")
    print("        " + "  ".join(f"{k}={metrics[k]}" for k in
          ["accuracy", "precision", "recall", "f1", "roc_auc"]))
    print("\n" + classification_report(y_te, p_te,
          target_names=["not-candidate", "retrofit-to-C"]))

    # ---- optional: skore report (nice-to-have, never fatal) ------------------
    try:
        from skore import EstimatorReport
        rep = EstimatorReport(model, X_train=X_tr, y_train=y_tr,
                              X_test=X_te, y_test=y_te, pos_label=1)
        summary = rep.metrics.summarize().frame()
        summary.to_csv(os.path.join(outdir, "skore_metrics.csv"))
        print("[skore] evaluation report written -> skore_metrics.csv")
    except Exception as e:                       # portability > completeness
        print(f"[skore] skipped ({type(e).__name__}); sklearn metrics used.")

    # ---- explainability exports ----------------------------------------------
    feat_names = _feature_names(model, cat, num)
    imp = pd.DataFrame({"feature": feat_names,
                        "importance": model.named_steps["tree"].feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
    print("\n[explain] top drivers of retrofit-to-C priority:")
    for _, r in imp.head(8).iterrows():
        if r.importance > 0:
            print(f"          {r.importance:6.3f}  {r.feature}")

    # human-readable rules
    rules = export_text(model.named_steps["tree"], feature_names=list(feat_names))
    with open(os.path.join(outdir, "decision_tree_rules.txt"), "w") as fh:
        fh.write("RETROFIT-TO-C  DECISION RULES  (read top to bottom)\n")
        fh.write("class 1 = retrofit-to-C candidate\n" + "=" * 60 + "\n")
        fh.write(rules)

    # the tree as a picture
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(26, 12))
        plot_tree(model.named_steps["tree"], feature_names=list(feat_names),
                  class_names=["not-candidate", "retrofit-to-C"], filled=True,
                  rounded=True, fontsize=8, ax=ax, impurity=False, proportion=True)
        fig.savefig(os.path.join(outdir, "decision_tree.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("[explain] tree diagram -> decision_tree.png")
    except Exception as e:
        print(f"[explain] tree plot skipped ({type(e).__name__})")

    with open(os.path.join(outdir, "model_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    return model, metrics


def _feature_names(model, cat, num):
    """Recover readable feature names after one-hot encoding."""
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat))
    return np.array(cat_names + num)


# ------------------------------------------------------------------------------
# 5. PRIORITISE  (Layers 1, 3, 4)  -- transparent, rule-based
# ------------------------------------------------------------------------------
def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def score_priority(df):
    """
    Turn the funnel into a single, auditable 0-100 priority score plus the
    persona routing every stakeholder needs. No ML here -- pure documented rules.
    """
    df = df.copy()

    # --- NEED: distance below EPC C (0 if already at/above C) -----------------
    rank = {r: i for i, r in enumerate(EPC_ORDER)}          # F-G=0 ... A-B=4
    target_rank = rank[TARGET_RATING]
    df["need_raw"] = (target_rank - df["epc_rating"].map(rank)).clip(lower=0)

    # --- BENEFIT: achievable EPC uplift ---------------------------------------
    df["benefit_raw"] = df["epc_uplift"]

    # --- EQUITY: fuel poverty + income deprivation ----------------------------
    #     income decile 1 = most deprived -> invert so high == more deprived
    df["equity_raw"] = (
        _minmax(df["fuel_poverty"]) +
        _minmax(10 - df["imd19_income_decile"])
    ) / 2.0

    df["priority_score"] = 100 * (
        W_NEED    * _minmax(df["need_raw"]) +
        W_BENEFIT * _minmax(df["benefit_raw"]) +
        W_EQUITY  * df["equity_raw"]
    )
    # properties that already meet the target get zero priority for retrofit
    df.loc[df["need_state"] == "already_efficient", "priority_score"] = 0.0

    # --- TIER: quartile-style bands among homes that need work ----------------
    needs = df["need_state"] != "already_efficient"
    df["priority_tier"] = "Tier 4 - none/monitor"
    if needs.sum() > 0:
        q = df.loc[needs, "priority_score"]
        t1, t2, t3 = q.quantile([0.90, 0.70, 0.40])
        band = pd.Series("Tier 3 - lower", index=df.index)
        band[needs & (df.priority_score >= t3)] = "Tier 2 - medium"
        band[needs & (df.priority_score >= t2)] = "Tier 1 - high"
        band[needs & (df.priority_score >= t1)] = "Tier 0 - urgent"
        df["priority_tier"] = np.where(needs, band, "Tier 4 - none/monitor")

    # --- EQUITY SEGMENT: can they self-fund, or need public help? -------------
    fp_hi = df["fuel_poverty"] >= df["fuel_poverty"].median()
    dep   = df["imd19_income_decile"] <= 4          # more-deprived neighbourhoods
    support_needed = fp_hi | dep
    df["funding_route"] = np.select(
        [
            df["need_state"] == "already_efficient",
            df["tenure"].eq("social"),
            df["tenure"].isin(["private-rented", "privately rented"]),
            (df["tenure"] == "owner-occupied") & support_needed,
            (df["tenure"] == "owner-occupied") & ~support_needed,
        ],
        [
            "n/a - already efficient",
            "Public / social-landlord programme",
            "Landlord obligation (MEES) + support",
            "Grant-supported owner (ECO/local)",
            "Able-to-pay owner (self-fund + nudge)",
        ],
        default="Review case-by-case",
    )

    # --- PERSONA OWNER: who acts first ----------------------------------------
    df["persona_owner"] = np.select(
        [
            df["need_state"] == "already_efficient",
            df["funding_route"].str.startswith("Able-to-pay"),
            df["funding_route"].str.startswith("Landlord"),
            df["funding_route"].str.startswith("Grant") | df["funding_route"].str.startswith("Public"),
            df["need_state"] == "deep_retrofit",
        ],
        [
            "Resident (maintain)",
            "Resident (self-fund) - Council nudges",
            "Corporation/Landlord - Council enforces",
            "Council (target public funds) + GLA",
            "GLA (strategic deep-retrofit programme)",
        ],
        default="Council",
    )

    # --- ACTION: cheapest high-impact measure from the fabric flags -----------
    df["recommended_measure"] = _recommend_measure(df)

    # --- FEASIBILITY + CO-BENEFIT flags ---------------------------------------
    df["heritage_constraint"] = np.where(
        (df["is_listed"] == "listed") |
        (df["conservation_area_flag"] == "in conservation area"),
        "constrained (specialist approval)", "standard")
    df["solar_opportunity"] = pd.cut(
        df["solar_pv_potential"], bins=[-1, 0, 2000, 4000, 1e9],
        labels=["none", "modest", "good", "excellent"])
    df["heat_risk_cobenefit"] = np.where(
        df["heat_risk_quintile"] >= 4,
        "high overheating risk - specify cooling-friendly measures", "standard")

    return df


def _recommend_measure(df):
    """Rank the single most cost-effective first measure per property."""
    solid_uninsul = (df["wall_type"] == "solid") & (df["wall_insulation"] == "uninsulated")
    cavity_uninsul = (df["wall_type"] == "cavity") & (df["wall_insulation"] == "uninsulated")
    roof_uninsul = df["roof_insulation"] == "uninsulated"
    single_glaze = df["glazing_type"] == "single/partial"
    fossil_heat = df["main_fuel_type"].isin(["mains gas", "other"]) & \
                  ~df["main_heat_type"].isin(["heat pump"])

    return np.select(
        [cavity_uninsul, roof_uninsul, solid_uninsul, single_glaze, fossil_heat],
        [
            "Cavity wall insulation (quick, low-cost win)",
            "Loft / roof insulation (quick, low-cost win)",
            "Solid wall insulation (higher-cost, high-impact)",
            "Glazing upgrade to double/triple",
            "Low-carbon heating (heat pump) switch",
        ],
        default="Fabric already strong - review ventilation/controls",
    )


# ------------------------------------------------------------------------------
# 6. BOROUGH + NEIGHBOURHOOD SUMMARIES
# ------------------------------------------------------------------------------
def borough_summary(df):
    borough = df["administrative_area"].iloc[0] if "administrative_area" in df else "Unknown"
    total = len(df)
    ns = df["need_state"].value_counts()
    print("\n" + "=" * 62)
    print(f"BOROUGH SUMMARY:  {borough}")
    print("=" * 62)
    print(f"  Homes assessed (reliable data) : {total:,}")
    print(f"  Already meet EPC C             : {ns.get('already_efficient',0):,} "
          f"({ns.get('already_efficient',0)/total*100:.1f}%)")
    print(f"  Retrofit-to-C candidates       : {ns.get('retrofit_to_C',0):,} "
          f"({ns.get('retrofit_to_C',0)/total*100:.1f}%)")
    print(f"  Deep-retrofit (can't reach C)  : {ns.get('deep_retrofit',0):,} "
          f"({ns.get('deep_retrofit',0)/total*100:.1f}%)")
    urgent = (df["priority_tier"] == "Tier 0 - urgent").sum()
    print(f"  Tier-0 URGENT homes            : {urgent:,}")
    print("\n  Funding route split:")
    for k, v in df["funding_route"].value_counts().items():
        print(f"    - {k:42s} {v:6,}")
    print("\n  Top recommended first measures:")
    for k, v in df["recommended_measure"].value_counts().head(5).items():
        print(f"    - {k:48s} {v:6,}")
    return borough


def lsoa_summary(df, outdir):
    """Neighbourhood roll-up so councils can target streets, not just homes."""
    if "lsoa21nm" not in df.columns:
        return
    g = df.groupby("lsoa21nm").agg(
        homes=("uprn", "count"),
        candidates=("is_retrofit_candidate", "sum"),
        deep_retrofit=("need_state", lambda s: (s == "deep_retrofit").sum()),
        mean_priority=("priority_score", "mean"),
        mean_fuel_poverty=("fuel_poverty", "mean"),
        median_income_decile=("imd19_income_decile", "median"),
        tier0_urgent=("priority_tier", lambda s: (s == "Tier 0 - urgent").sum()),
    ).reset_index()
    g["candidate_rate_%"] = (g["candidates"] / g["homes"] * 100).round(1)
    g = g.sort_values("mean_priority", ascending=False)
    g.to_csv(os.path.join(outdir, "lsoa_summary.csv"), index=False)
    print(f"\n[export] neighbourhood roll-up -> lsoa_summary.csv "
          f"({len(g)} LSOAs)")
    print("\n  Top 5 priority neighbourhoods (by mean priority score):")
    for _, r in g.head(5).iterrows():
        print(f"    {r['lsoa21nm']:24s} score={r['mean_priority']:5.1f} "
              f"candidates={int(r['candidates']):4d} "
              f"fuel_pov={r['mean_fuel_poverty']:4.1f}%")


# ------------------------------------------------------------------------------
# 7. EXPORT PROPERTY-LEVEL RESULTS
# ------------------------------------------------------------------------------
def export_properties(df, outdir):
    keep = [c for c in [
        "uprn", "postcode_locator", "lsoa21nm", "ward22nm",
        "property_type_built_form", "tenure", "construction_age_band",
        "epc_rating", "potential_epc_rating", "epc_uplift", "energy_consumption",
        "need_state", "priority_score", "priority_tier",
        "funding_route", "persona_owner", "recommended_measure",
        "heritage_constraint", "solar_opportunity", "heat_risk_cobenefit",
    ] if c in df.columns]
    out = df[keep].sort_values("priority_score", ascending=False)
    out.to_csv(os.path.join(outdir, "property_scores.csv"), index=False)
    print(f"[export] per-property scores -> property_scores.csv "
          f"({len(out):,} homes)")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def run(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = load_borough(input_path)
    df = clean_and_filter(df)
    df = build_label(df)
    model, metrics = train_model(df, outdir)
    df = score_priority(df)
    borough_summary(df)
    lsoa_summary(df, outdir)
    export_properties(df, outdir)

    # save the trained model so it can be reused on other boroughs
    try:
        import joblib
        joblib.dump(model, os.path.join(outdir, "retrofit_model.joblib"))
        print("[export] trained model -> retrofit_model.joblib")
    except Exception:
        pass
    print(f"\nDone. All outputs in: {os.path.abspath(outdir)}")
    return df, model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Warmer Homes retrofit prioritiser")
    ap.add_argument("--input", required=True, help="Path to an LBSMv2 borough CSV")
    ap.add_argument("--outdir", default="./out", help="Output directory")
    args = ap.parse_args()
    run(args.input, args.outdir)
