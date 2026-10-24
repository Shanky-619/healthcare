# QualityLens

**Explainable hybrid-ensemble prediction and discrepancy audit of CMS hospital star ratings**

QualityLens predicts the CMS Overall Hospital Quality Star Rating (1–5) from the measure-group counts that CMS publishes for each hospital, explains which inputs drive the prediction, and flags hospitals whose official rating is out of line with their own measure profile. The flag is the **Quality Rating Discrepancy Index (QRDI)**.

> ⚠️ **Dataset check.** The course requires a dataset from the instructor's Moodle list. Moodle item 6 (`7t8x-u3ir`) is *Inpatient Rehabilitation Facility – General Information*, which has no rating column. This project uses *Hospital General Information*, which the Moodle "Comprehensive Lists" catalogue (HealthData.gov) lists as [`n7m7-8qpq`](https://healthdata.gov/d/n7m7-8qpq) and which links to the CMS source file [`xubh-q36u`](https://data.cms.gov/provider-data/dataset/xubh-q36u). Confirm with the instructor that this is acceptable before you submit. The dataset ID is a single setting in `src/qualitylens/config.py`.

---

## Dataset

| | |
|---|---|
| Source | CMS Provider Data Catalog, *Hospital General Information* |
| Page | https://data.cms.gov/provider-data/dataset/xubh-q36u |
| API | `https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0?limit=500&offset=N` |
| Unit | one row per Medicare-registered hospital: 5,419 rows × 38 columns on 2026-10-15; 3,174 rated, 2,245 "Not Available" |
| Target | `hospital_overall_rating` (1–5; "Not Available" rows are excluded) |
| Refresh | quarterly. `data/raw/provenance.json` records the download date, row count and SHA-256 |

The snapshot used for the reported run is committed as `data/raw/hospital_general_information.csv` (retrieved 2026-10-15, SHA-256 in `data/raw/provenance.json`). CMS refreshes the file quarterly, so a new download may give slightly different numbers. To get the latest version:
1. `python -m qualitylens fetch --refresh` (needs internet access to data.cms.gov), or
2. download the CSV from the dataset page and save it as `data/raw/hospital_general_information.csv`.

## Approach

```
raw CSV/API ─► cleaning report ─► features ─► models ─► evaluation ─► explanation ─► QRDI audit ─► dashboard
                 ("Not Available",   raw counts     5 baselines     8 metrics +     permutation     out-of-fold
                  duplicates,        + engineered   tuned RF/HGB    ordinal metrics  importance,     expected rating,
                  unrated dropped)   + context      stacking        bootstrap CIs    SHAP            group tests
```

**What is new compared with a basic EDA plus classifier:**
1. **Engineered net-performance features.** For each outcome group, (better − worse) / reported, plus coverage and aggregates. An ablation study measures how much they add.
2. **Ordinal-aware stacking.** A tuned random forest and a tuned gradient-boosting model are blended by a logistic meta-learner. Tuning is scored with quadratic-weighted kappa, and an ordinal decoding step rounds the expected rating. It is compared with five baselines using paired bootstrap confidence intervals.
3. **QRDI audit.** A model trained on performance features only gives each hospital an out-of-fold expected rating. QRDI = official − expected. QRDI is then compared across ownership, hospital type and state with Kruskal–Wallis tests and ε² effect sizes.

4. **Quality archetypes + prescriptive what-if.** Unsupervised k-means archetypes, with k chosen by silhouette and Davies–Bouldin and shown on a PCA map, plus a greedy per-hospital "path to the next star" search over realistic one-measure changes, built on the trained model.

**Limitation (state it in the report).** CMS computes the star rating from these same measure groups using continuous scores that this file does not contain. QRDI therefore screens for internal consistency. It does not prove that a hospital was mis-rated.

## Repository layout

```
src/qualitylens/   config · data · features · models · evaluate · explain · audit · segment · prescribe · viz · pipeline · synthetic
app/               dashboard.py (Streamlit, 9 pages) + theme.py (colours, CSS, Plotly template)
report/            build_report.py → QualityLens_Report_Template.docx (B/W, all 19 sections, tables/figures/diagrams filled)
.streamlit/        dashboard theme
notebooks/         QualityLens.ipynb – step-by-step walkthrough with interpretation cells
presentation/      build_deck.py → QualityLens_Presentation.pptx (numbers read from results/), screens/
tests/             unit + smoke tests (pytest)
docs/              report guide, viva prep, presentation outline, results fact sheet
data/raw/          downloaded CSV + provenance.json (git-ignored CSV)
results/           outputs of a real run: figures/, tables/, metrics.json, run_info.json, SUMMARY.md
legacy/            first single-notebook prototype (superseded)
```

## Dashboard

`streamlit run app/dashboard.py` opens nine pages:

| Page | What it shows |
|---|---|
| Overview | KPIs, star distribution, US state tile map (mean stars / share 4–5★ / mean QRDI), star mix by ownership, type, emergency services or birthing-friendly status, engineered-feature box plot, data completeness |
| Key insights | eight findings computed live from the results (ownership, readmissions, type, peer group, engineered signal, prescriptive, geography, audit), each with a mini chart |
| Model performance | headline metrics with CIs, interactive model comparison (κ / F1 / accuracy), full metric table, confusion matrix, ROC curves, **calibration (reliability diagram, ECE, Brier)**, per-class report, CV, ablation |
| Explainability | permutation importance, mean \|SHAP\|, SHAP-by-class heatmap, agreement check |
| Rating audit (QRDI) | distribution, expected vs official, group comparison with Kruskal–Wallis and ε², state map, largest discrepancies |
| Quality archetypes | k-means segmentation (silhouette and Davies–Bouldin for k = 2…8), PCA map of all hospitals, archetype profiles and star mix |
| Hospital explorer | profile card, probability profile, position vs national distribution, plus three tabs: **why this rating** (occlusion explanation), **path to the next star** (prescriptive what-if), **peer twins** (nearest neighbours) |
| What-if simulator | set measure results (or start from a real hospital) and get the model's star distribution |
| Method & data | pipeline, provenance, cleaning report, novelty, library versions |

Colours: gold→bronze for star ratings (ordinal), wine for the proposed model, plum/copper for the sign of QRDI. All three sets were checked for colour-blind separation and contrast.

## Presentation

`python presentation/build_deck.py` rebuilds `presentation/QualityLens_Presentation.pptx` (21 slides) from `results/`. Slide text is short and factual. Speaker notes are prompts, not a script: say it in your own words.

## Poster

`python poster/poster_figs.py` draws poster-sized charts, `python poster/build_poster.py` writes `poster/poster.html` in the dashboard's visual style, and `python poster/render_poster.py` prints `QualityLens_Poster_A3.pdf` and `QualityLens_Poster_A4.pdf` (plus PNG previews) with headless Chrome. A4 is the same page scaled by 1/√2. Every number is read from `results/`.

## Report template

`python report/build_report.py` builds `report/QualityLens_Report_Template.docx`, a black-and-white Word document with all 19 sections the brief requires. Every table of facts and numbers, every figure, four method diagrams, the formulas and a reference list are filled in from `results/`. Each section has a dashed "write this in your own words" box listing what to cover. **The prose is deliberately left to the student**, because the brief requires an original, AI-free report. Delete each box after writing the section. After opening in Word, right-click the table of contents and choose *Update field*.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python -m qualitylens fetch                 # download + provenance
python -m qualitylens run --fast --out results_fast   # quick check (~8 min on 2 cores)
python -m qualitylens run                   # full run: tuning, 1000 bootstraps, 5-fold CV (~26 min on 2 cores)
python -m qualitylens summary               # rebuild results/SUMMARY.md + importance figures, no retraining
streamlit run app/dashboard.py              # dashboard reads results/ (open http://localhost:8501)
python presentation/build_deck.py           # rebuild the slide deck from results/
pytest                                      # tests
```

The committed `results/` folder holds the real run of 2026-10-15, including the trained model (`model.joblib`, ~10 MB, saved with scikit-learn 1.9.1). If your scikit-learn version cannot load it, the what-if page says so; one `python -m qualitylens run` recreates it. Every other page works straight from the committed tables.

Google Colab: upload the repo (or `git clone` it), then run `!pip install -e .`, followed by the commands above with a leading `!`.

`python -m qualitylens run --demo` runs on **synthetic** data that has the real column layout. Use it only to check that the code works. Every figure from it is watermarked, and its numbers must never appear in the report.

Full runtime depends on the CPU. On the 2-core laptop used for the reported run, `--fast` took about 8 minutes and the full run about 26.

## Outputs (in `results/`)

| File | Content |
|---|---|
| `tables/model_comparison.csv` | accuracy, macro precision/recall/specificity/F1, weighted F1, QWK, MAE, within-one-star and ROC-AUC for every model |
| `tables/bootstrap_ci.csv` | 95% CIs, plus the paired difference against the best baseline |
| `tables/ablation.csv` | contribution of each feature group |
| `tables/per_class_proposed.csv`, `confusion_proposed.csv` | per-class sensitivity/specificity and the confusion matrix |
| `tables/permutation_importance.csv`, `shap_importance.csv` | explanations |
| `tables/qrdi_*.csv` | QRDI for every hospital, by group, and top outliers |
| `tables/hospital_profiles.csv` | one row per rated hospital: engineered features, expected stars, QRDI, probability profile |
| `tables/roc_curves_proposed.csv` | ROC points per star class (dashboard) |
| `tables/cluster_selection.csv`, `archetype_profiles.csv`, `archetype_zscores.csv` | archetype selection and profiles |
| `tables/next_star_first_move.csv`, `prescriptive_moves.csv` | best single move per hospital and its summary |
| `tables/calibration.csv`, `test_predictions.csv` | reliability bins; test-set predictions with probabilities |
| `figures/fig01…fig22` | all figures used in the report (fig16/17 state maps, fig18–20 archetypes, fig21 prescriptive, fig22 calibration) |
| `metrics.json`, `run_info.json`, `SUMMARY.md` | machine-readable results, library versions and provenance |

## Tools and versions

Python ≥ 3.10 · pandas · NumPy · scikit-learn · SciPy · matplotlib · SHAP · Streamlit · pytest. `run_info.json` records the exact versions of every run; copy them into the report from there.

## Reproducibility

Fixed seed (42), a stratified 75/25 split, and all preprocessing inside sklearn Pipelines, so no test data leaks into training. The QRDI uses out-of-fold predictions only. Provenance and versions are written alongside every run.
