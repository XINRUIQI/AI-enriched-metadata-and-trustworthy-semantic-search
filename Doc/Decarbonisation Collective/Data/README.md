# Warmer Homes — Retrofit Prioritisation Engine (MVP)

An interpretable pipeline over the London Building Stock Model v2 (LBSMv2) that
categorises every home in a borough into a **priority tier**, a **funding route**,
a **recommended first measure**, and a **persona owner**.

## Run it

```bash
pip install scikit-learn pandas numpy matplotlib joblib
pip install skore          # optional — nicer eval report; script runs fine without it

python warmer_homes_model.py \
    --input LBSMv2_Hammersmith_Fulham.csv \
    --outdir ./out
```

## Plug in another borough

Just swap the `--input` file for any borough's LBSMv2 CSV (same 63-column schema):

```bash
python warmer_homes_model.py --input LBSMv2_Camden.csv --outdir ./out_camden
```

Each borough gets its **own** explainable model and its own output bundle, so the
logic stays true to local building stock.

## Outputs (in `--outdir`)

| File | What it is | Who it's for |
|---|---|---|
| `property_scores.csv` | one row per home: tier, funding route, measure, owner | everyone |
| `lsoa_summary.csv` | neighbourhood roll-up (target streets, not just homes) | councillors, GLA |
| `decision_tree.png` | the model, as a picture | analysts, presentation |
| `decision_tree_rules.txt` | the model, as IF/THEN rules | analysts, audit |
| `feature_importance.csv` | what physically drives priority | analysts |
| `model_metrics.json` | accuracy / precision / recall / ROC-AUC | analysts |
| `skore_metrics.csv` | skore evaluation report (if installed) | analysts |
| `retrofit_model.joblib` | the trained model, reusable | engineers |
| `funnel_concept.png` | the layered-funnel concept diagram | presentation |
| `PRESENTATION.md` | the write-up of the idea + results | presentation |

## Key design choices (all in the CONFIG block at the top of the script)

- **Data gate:** residential homes with every `*_known` flag = 1 (no guessed attributes).
- **Target:** *retrofit-to-C candidate* = below EPC C now **and** can reach C with standard
  measures. `deep_retrofit` (can't reach C) is split out for a strategic GLA track.
- **Model:** one shallow decision tree — interpretability over raw accuracy, on purpose.
  Current/potential EPC and energy consumption are **excluded** from the model to prevent
  leakage and to keep the tree fabric-driven and actionable.
- **Priority score:** a documented, weighted blend of **need + benefit + equity** — no
  black box. Weights live in `W_NEED`, `W_BENEFIT`, `W_EQUITY`.

## Problematic columns — how they're handled (simple + safe)

- `basement_floor` (~90% NaN) → NaN means "no basement" → `0`
- `solar_pv_area` / `solar_pv_potential` / `avg_tilt` (NaN = no viable roof) → `0`
- `estimated_floor_count` (small NaN) → median
- `listed_building_grade` → collapsed to a binary `is_listed` flag
- IDs / coordinates / high-cardinality codes (`os_topo_toid`, `easting`, `northing`,
  `loac_group`, `conservation_area_site_id`) → dropped from modelling
