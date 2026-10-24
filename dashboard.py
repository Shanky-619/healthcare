"""QualityLens dashboard.   streamlit run app/dashboard.py

Reads the outputs written by `python -m qualitylens run` (default folder: results/).
Nothing here re-trains a model: every number comes from the saved run, and the
what-if simulator uses the saved stacking model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

import theme as T  # noqa: E402
from qualitylens import config as C  # noqa: E402
from qualitylens import features as F  # noqa: E402
from qualitylens import prescribe as PR  # noqa: E402
from qualitylens import segment as SG  # noqa: E402
from qualitylens.viz import STATE_TILES  # noqa: E402

st.set_page_config(page_title="QualityLens · CMS Star-Rating Intelligence", page_icon="★", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.CSS, unsafe_allow_html=True)

_NEW_WIDTH = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 46)
WIDE = {"width": "stretch"} if _NEW_WIDTH else {"use_container_width": True}
PCFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

PAGES = ["Overview", "Key insights", "Model performance", "Explainability", "Rating audit (QRDI)", "Quality archetypes",
         "Hospital explorer", "What-if simulator", "Method & data"]
MODEL_SHORT = {
    "Majority class": "Majority class",
    "Logistic regression": "Logistic regression",
    "Decision tree": "Decision tree",
    "Random forest": "Random forest",
    "Gradient boosting": "Gradient boosting",
    "Random forest (tuned)": "Random forest (tuned)",
    "Gradient boosting (tuned)": "Gradient boosting (tuned)",
    "Stacking ensemble (proposed)": "Stacking ensemble ★",
    "Stacking + ordinal decoding (proposed)": "Stacking + ordinal decoding ★",
}
METRIC_LABEL = {
    "accuracy": "Accuracy", "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
    "specificity_macro": "Specificity (macro)", "f1_macro": "F1 (macro)", "f1_weighted": "F1 (weighted)",
    "qwk": "Quadratic-weighted κ", "mae_stars": "MAE (stars)", "within_one_star": "Within ±1 star",
    "roc_auc_ovr_macro": "ROC-AUC (OvR macro)",
}
DOMAIN_LABELS = C.DOMAIN_LABELS
OWN_SHORT = {
    "Voluntary non-profit - Private": "Non-profit · private",
    "Voluntary non-profit - Other": "Non-profit · other",
    "Voluntary non-profit - Church": "Non-profit · church",
    "Proprietary": "For-profit (proprietary)",
    "Government - Hospital District or Authority": "Gov · hospital district",
    "Government - Local": "Gov · local",
    "Government - State": "Gov · state",
    "Government - Federal": "Gov · federal",
    "Veterans Health Administration": "Veterans Health Admin.",
    "Physician": "Physician-owned",
    "Tribal": "Tribal",
    "Department of Defense": "Dept. of Defense",
}


# --------------------------------------------------------------------------- helpers
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, note, accent)"""
    cells = "".join(
        f'<div class="ql-kpi {acc}"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="note">{note}</div></div>'
        for lbl, val, note, acc in items)
    html(f'<div class="ql-kpis">{cells}</div>')


def section(title, caption=None):
    html(f'<div class="ql-section">{title}</div>' + (f'<div class="ql-caption">{caption}</div>' if caption else ""))


def plot(fig, height=360, key=None):
    fig.update_layout(height=height)
    st.plotly_chart(fig, config=PCFG, key=key, **WIDE)


def table(df, **kw):
    st.dataframe(df, hide_index=True, **WIDE, **kw)


def stars_html(k: int) -> str:
    k = int(k)
    return f'<span class="ql-stars">{"★" * k}<span class="off">{"★" * (5 - k)}</span></span>'


def flag_badge(flag: str) -> str:
    cls = {"rated above profile": "above", "rated below profile": "below"}.get(flag, "ok")
    return f'<span class="ql-badge {cls}">{flag}</span>'


pretty = F.pretty_name


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load(folder: str):
    R = ROOT / folder
    out = {"metrics": json.loads((R / "metrics.json").read_text()),
           "info": json.loads((R / "run_info.json").read_text())}
    for p in (R / "tables").glob("*.csv"):
        out[p.stem] = pd.read_csv(p)
    return out


@st.cache_resource(show_spinner=False)
def load_bundle(folder: str):
    return joblib.load(ROOT / folder / "model.joblib")


folders = [p.name for p in sorted(ROOT.glob("results*")) if (p / "metrics.json").exists()]
if not folders:
    st.error("No results found. Run `python -m qualitylens run` first (or `--demo` for a synthetic preview).")
    st.stop()
default_folder = folders.index("results") if "results" in folders else 0

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    html('<div class="ql-brand">Quality<span>Lens</span></div>'
         '<div class="ql-tag">CMS hospital star-rating intelligence</div><hr>')
    # ?page=audit etc. opens a page directly (handy for links and screenshots)
    slug = str(st.query_params.get("page", "")).lower()
    start_idx = next((i for i, p_ in enumerate(PAGES) if slug and p_.lower().replace("-", "").startswith(slug)), 0)
    page = st.radio("Navigate", PAGES, index=start_idx, label_visibility="collapsed", key="page")
    html("<hr>")
    folder = st.selectbox("Results folder", folders, index=default_folder)

D = load(folder)
M, INFO = D["metrics"], D["info"]
CL = M["cleaning"]
PROV = INFO.get("provenance", {})
SYN = bool(M.get("synthetic"))
TM = pd.DataFrame(M["test_metrics"]).T
PROP, PROP_ORD = "Stacking ensemble (proposed)", "Stacking + ordinal decoding (proposed)"
REF = M["best_baseline"]
H = D.get("hospital_profiles")
Q = D["qrdi_all_hospitals"]

with st.sidebar:
    retrieved = str(PROV.get("retrieved_utc", "—"))[:10]
    html(f'<div class="ql-side-card"><b>Dataset</b><br>{C.DATASET_NAME}<br>CMS Provider Data · <code>{C.DATASET_ID}</code><br>'
         f'Retrieved {retrieved} · {CL["rows_raw"]:,} rows<br><br><b>Run</b><br>seed {INFO["seed"]} · '
         f'{INFO["runtime_seconds"]:.0f}s · {"fast" if INFO["args"].get("fast") else "full"} mode</div>')

if SYN:
    st.error("SYNTHETIC DEMO DATA - these numbers are generated and are NOT results.")


def hero(eyebrow, title, sub, chips=()):
    ch = "".join(f'<span class="ql-chip {c[1] if isinstance(c, tuple) else ""}">{c[0] if isinstance(c, tuple) else c}</span>'
                 for c in chips)
    html(f'<div class="ql-hero"><div class="ql-eyebrow">{eyebrow}</div><div class="ql-title">{title}</div>'
         f'<div class="ql-sub">{sub}</div><div class="ql-chips">{ch}</div></div>')


def ci_row(metric, model):
    ci = D["bootstrap_ci"]
    r = ci[(ci.metric == metric) & (ci.model == model)]
    return r.iloc[0] if len(r) else None


# --------------------------------------------------------------------------- chart builders
def star_bar(counts: pd.Series, title=""):
    tot = counts.sum()
    fig = go.Figure(go.Bar(
        x=[f"{k}★" for k in counts.index], y=counts.values, marker=dict(color=T.STARS, line=dict(width=0)),
        text=[f"{v:,}<br><span style='font-size:11px'>{v / tot:.0%}</span>" for v in counts.values],
        textposition="outside", textfont=dict(color=T.INK2),
        hovertemplate="%{x}: %{y:,} hospitals<extra></extra>", cliponaxis=False))
    fig.update_layout(title=title, yaxis_title="Hospitals", showlegend=False, yaxis_range=[0, counts.max() * 1.25])
    fig.update_traces(marker_cornerradius=6)
    return fig


def tile_map(values: pd.Series, counts: pd.Series, diverging=False, fmt="{:.2f}", label="", min_n=5, extra=None):
    shown = values[(counts.reindex(values.index).fillna(0) >= min_n) & values.index.isin(STATE_TILES)]
    lo, hi = float(shown.min()), float(shown.max())
    lim = max(abs(lo), abs(hi), 1e-6)
    fig = go.Figure()
    xs, ys, txt, hov, tcol = [], [], [], [], []
    for stc, (cx, cy) in STATE_TILES.items():
        x, y = cx, -cy
        if stc in shown.index:
            v = shown[stc]
            col = (T.lerp([T.COPPER, T.MID, T.PLUM], (v + lim) / (2 * lim)) if diverging
                   else T.lerp([T.GOLD_LIGHT] + T.STARS, (v - lo) / (hi - lo or 1)))
            line = T.CARD
            tc = "#FFFFFF" if T.is_dark(col) else T.INK
            txt.append(f"<b>{stc}</b><br><span style='font-size:10px'>{fmt.format(v)}</span>")
            h = f"<b>{stc}</b><br>{label}: {fmt.format(v)}<br>Rated hospitals: {int(counts.get(stc, 0))}"
            if extra is not None and stc in extra.index:
                h += f"<br>{extra[stc]}"
            hov.append(h)
        else:
            col, line, tc = T.CARD, T.AXIS, T.MUTED
            txt.append(stc)
            hov.append(f"<b>{stc}</b><br>fewer than {min_n} rated hospitals")
        tcol.append(tc)
        fig.add_shape(type="rect", x0=x + 0.05, x1=x + 0.95, y0=y + 0.05, y1=y + 0.95, fillcolor=col,
                      line=dict(color=line, width=1.5), layer="below")
        xs.append(x + 0.5); ys.append(y + 0.5)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="text+markers", text=txt, textfont=dict(color=tcol, size=11),
                             marker=dict(size=34, symbol="square", color="rgba(0,0,0,0)"),
                             hovertext=hov, hoverinfo="text", showlegend=False))
    # colour key (legend for the ramp)
    if diverging:
        scale = [[0, T.COPPER], [0.5, T.MID], [1, T.PLUM]]; cmin, cmax = -lim, lim
    else:
        scale = [[i / 5, c] for i, c in enumerate([T.GOLD_LIGHT] + T.STARS)]; cmin, cmax = lo, hi
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=scale, cmin=cmin, cmax=cmax, color=[cmin], showscale=True,
                                         colorbar=dict(title=dict(text=label, side="right"), thickness=10, len=0.6, x=1.0,
                                                       outlinewidth=0, tickfont=dict(color=T.INK2)))))
    fig.update_xaxes(visible=False, range=[-0.1, 12.1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-7.1, 1.05], scaleanchor="x", fixedrange=True)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def mix_bars(df: pd.DataFrame, col: str, top=10):
    keep = df[col].value_counts().head(top).index
    d = df[df[col].isin(keep)]
    ct = pd.crosstab(d[col], d[C.TARGET], normalize="index").reindex(columns=C.CLASSES, fill_value=0)
    n = d[col].value_counts()
    ct["_hi"] = ct[4] + ct[5]
    ct = ct.sort_values("_hi")
    labels = [f"{OWN_SHORT.get(i, i)}  (n={n[i]:,})" for i in ct.index]
    fig = go.Figure()
    for k, c in zip(C.CLASSES, T.STARS):
        fig.add_trace(go.Bar(y=labels, x=ct[k], orientation="h", name=f"{k}★", marker=dict(color=c, line=dict(color=T.CARD, width=2)),
                             hovertemplate=f"%{{y}}<br>{k}★: %{{x:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1], title="Share of rated hospitals"),
                      legend=dict(orientation="h", y=1.08, x=0, title=None, traceorder="normal"), bargap=0.32)
    return fig


def box_by_star(values: pd.Series, stars: pd.Series, ytitle, diag=False):
    fig = go.Figure()
    for k, c in zip(C.CLASSES, T.STARS):
        v = values[stars == k].dropna()
        fig.add_trace(go.Box(y=v, name=f"{k}★", marker=dict(color=c, size=3, opacity=.45), line=dict(color=c, width=1.6),
                             fillcolor=T.lerp([T.CARD, c], .35), boxpoints="outliers", hoverinfo="y", showlegend=False))
    if diag:
        fig.add_trace(go.Scatter(x=[f"{k}★" for k in C.CLASSES], y=C.CLASSES, mode="lines+markers", name="perfect agreement",
                                 line=dict(color=T.MUTED, dash="dot", width=1.5), marker=dict(size=6, color=T.MUTED)))
    fig.update_layout(yaxis_title=ytitle, showlegend=diag, legend=dict(orientation="h", y=1.1, x=0))
    return fig


def importance_chart(names, vals, errs=None, color=T.WINE, xtitle=""):
    fig = go.Figure(go.Bar(y=names[::-1], x=vals[::-1], orientation="h", marker=dict(color=color, cornerradius=5),
                           error_x=None if errs is None else dict(array=errs[::-1], color=T.INK2, thickness=1.2, width=3),
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(xaxis_title=xtitle, bargap=0.35)
    return fig


def arche_color(a: int, k: int) -> str:
    """Archetypes are ranked by mean stars, so they reuse the ordinal star ramp."""
    return T.lerp(T.STARS, (a - 1) / max(k - 1, 1))


def safe_bundle():
    if not (ROOT / folder / "model.joblib").exists():
        return None
    try:
        return load_bundle(folder)
    except Exception:
        return None


GROUPS = {
    "Mortality results": [f"count_of_mort_measures_{k}" for k in ("better", "worse", "no_different")] + ["count_of_facility_mort_measures"],
    "Safety results": [f"count_of_safety_measures_{k}" for k in ("better", "worse", "no_different")] + ["count_of_facility_safety_measures"],
    "Readmission results": [f"count_of_readm_measures_{k}" for k in ("better", "worse", "no_different")] + ["count_of_facility_readm_measures"],
    "Patient-experience reporting": ["count_of_facility_pt_exp_measures"],
    "Timely & effective reporting": ["count_of_facility_te_measures"],
    "Hospital type": ["hospital_type"], "Ownership": ["hospital_ownership"], "State": ["state"],
    "Emergency services": ["emergency_services"], "Birthing-friendly": ["birthing_friendly"],
}


def local_explanation(bundle, row: dict, typical: dict) -> pd.DataFrame:
    """Occlusion: replace one group of inputs with the national typical value and see how far the
    expected rating moves. Positive = this group lifts the hospital above a typical hospital."""
    variants = [row]
    names = []
    for g, cols in GROUPS.items():
        v = dict(row)
        for c in cols:
            if c in typical:
                v[c] = typical[c]
        variants.append(v)
        names.append(g)
    exp, _, _ = PR.score(bundle["model"], variants, bundle["num_cols"], bundle["cat_cols"])
    return pd.DataFrame({"group": names, "contribution": exp[0] - exp[1:]}).sort_values("contribution")


@st.cache_resource(show_spinner=False)
def peer_index(folder_: str):
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Hh = load(folder_)["hospital_profiles"]
    cols = SG.profile_columns(Hh)
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Hh[cols])
    Z = prep.transform(Hh[cols])
    return NearestNeighbors(n_neighbors=6).fit(Z), Z


# ============================================================================ PAGES
if page == "Overview":
    hero("Healthcare analytics · CMS Provider Data Catalog",
         "What separates a 5-star hospital from a 1-star one?",
         "QualityLens predicts the CMS Overall Hospital Quality Star Rating from the measure-group results that CMS "
         "publishes for every Medicare-registered hospital, explains the drivers, and audits which hospitals carry an "
         "official rating that is out of line with their own performance profile.",
         [(f"{C.DATASET_NAME} · {C.DATASET_ID}", "wine"), f"Retrieved {retrieved}", "Ordinal 5-class target",
          "Explainable stacking ensemble", "QRDI audit"])
    unr = CL.get("rows_without_rating", CL["rows_raw"] - CL["rows_clean"])
    mean_star = float(H[C.TARGET].mean()) if H is not None else float(Q.official_stars.mean())
    kpis([("Hospitals in file", f"{CL['rows_raw']:,}", "Medicare-registered, one row each", ""),
          ("Rated & modelled", f"{CL['rows_clean']:,}", f"{unr:,} unrated rows excluded", "wine"),
          ("Mean star rating", f"{mean_star:.2f}★", "across rated hospitals", ""),
          ("States & territories", f"{Q['state'].nunique()}", "with at least one rated hospital", "plum"),
          ("Training hospitals", f"{M['n_train']:,}", f"+ {M['n_test']:,} held out for testing (stratified)", "copper")])

    c1, c2 = st.columns([1, 1.5])
    with c1:
        section("How the stars are distributed", "Rated hospitals only. Most hospitals sit in the middle of the scale.")
        cc = Q.official_stars.value_counts().reindex(C.CLASSES).fillna(0).astype(int)
        plot(star_bar(cc), 452)
    with c2:
        section("Geography of quality", "One tile per state. Hover for the number of hospitals.")
        mode = st.segmented_control("Map metric", ["Mean stars", "Share 4–5★", "Mean QRDI"], default="Mean stars",
                                    label_visibility="collapsed") if hasattr(st, "segmented_control") else "Mean stars"
        n_state = Q.groupby("state").size()
        if mode == "Share 4–5★":
            v = Q.assign(hi=Q.official_stars >= 4).groupby("state")["hi"].mean()
            plot(tile_map(v, n_state, fmt="{:.0%}", label="Share 4–5★"), 400)
        elif mode == "Mean QRDI":
            v = Q.groupby("state")["qrdi"].mean()
            plot(tile_map(v, n_state, diverging=True, fmt="{:+.2f}", label="Mean QRDI"), 400)
        else:
            v = Q.groupby("state")["official_stars"].mean()
            plot(tile_map(v, n_state, fmt="{:.2f}", label="Mean stars"), 400)

    section("Star mix by hospital characteristic", "Sorted by the share of 4- and 5-star hospitals.")
    dims = {"Ownership": "hospital_ownership", "Hospital type": "hospital_type", "Emergency services": "emergency_services",
            "Birthing-friendly designation": "birthing_friendly"}
    dsel = st.selectbox("Group by", list(dims), key="mix_dim")
    plot(mix_bars(Q.rename(columns={"official_stars": C.TARGET}), dims[dsel]), 120 + 38 * min(Q[dims[dsel]].nunique(), 10))

    if H is not None and "mean_net_performance" in H:
        c1, c2 = st.columns(2)
        with c1:
            section("The engineered signal tracks the stars",
                    "Mean net performance = average over outcome groups of (better − worse) / reported measures.")
            plot(box_by_star(H["mean_net_performance"], H[C.TARGET], "Mean net performance"), 380)
        with c2:
            section("Data completeness by measure group", "Share of rated hospitals reporting at least one measure.")
            cov = {DOMAIN_LABELS[d]: float((H[f"count_of_facility_{d}_measures"].fillna(0) > 0).mean())
                   for d in C.DOMAINS if f"count_of_facility_{d}_measures" in H}
            s = pd.Series(cov).sort_values()
            fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker=dict(color=T.WINE, cornerradius=5),
                                   text=[f"{v:.0%}" for v in s.values], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_layout(xaxis=dict(range=[0, 1.12], tickformat=".0%"), bargap=0.4)
            plot(fig, 380)

# ---------------------------------------------------------------------------
elif page == "Key insights":
    hero("Healthcare insights · computed from the run", "What the data says, in eight findings",
         "Each card is calculated live from the results tables, not typed by hand, so it always matches the data. "
         "Hover any chart for exact values.",
         [("Descriptive", "wine"), "Diagnostic", "Predictive", "Prescriptive"])
    if H is None:
        st.warning("This results folder has no hospital_profiles table.")
        st.stop()
    Hs = H.copy()
    Hs["hi"] = Hs[C.TARGET] >= 4

    def card(eyebrow, big, text, fig=None, height=210):
        with st.container(border=True):
            html(f'<div class="ql-eyebrow">{eyebrow}</div><div style="font-family:Fraunces,serif;font-size:2rem;'
                 f'font-weight:600;color:{T.INK};line-height:1.1;margin:.3rem 0 .2rem">{big}</div>'
                 f'<div style="color:{T.INK2};font-size:.92rem;line-height:1.5">{text}</div>')
            if fig is not None:
                fig.update_layout(margin=dict(l=8, r=16, t=8, b=8), showlegend=False)
                plot(fig, height)

    def hbar(labels, vals, colors, fmt=".0%"):
        f = go.Figure(go.Bar(y=labels, x=vals, orientation="h", marker=dict(color=colors, cornerradius=4),
                             text=[format(v, fmt) for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
        f.update_xaxes(visible=False, range=[min(0, min(vals)) * 1.3, max(vals) * 1.3 if max(vals) > 0 else 0.1])
        return f

    c1, c2 = st.columns(2)
    with c1:  # 1 ownership
        own = Hs.groupby("hospital_ownership").agg(n=("hi", "size"), hi=("hi", "mean")).query("n >= 30").sort_values("hi")
        top, bot = own.index[-1], own.index[0]
        card("Ownership", f"{own.hi.iloc[-1]:.0%} vs {own.hi.iloc[0]:.0%}",
             f"of <b>{OWN_SHORT.get(top, top)}</b> hospitals hold 4–5★, against {OWN_SHORT.get(bot, bot)} at the other end "
             f"(groups with ≥ 30 rated hospitals).",
             hbar([OWN_SHORT.get(i, i) for i in own.index], own.hi.values, [T.WINE if i in (top, bot) else T.GREY for i in own.index]), 250)
    with c2:  # 2 readmissions
        r = Hs.dropna(subset=["count_of_readm_measures_worse"])
        w = r.count_of_readm_measures_worse.clip(upper=3)
        g = r.groupby(w)[C.TARGET].mean()
        gap = g.iloc[0] - g.iloc[-1]
        card("Readmissions", f"{abs(gap):.2f}★",
             f"{'lower' if gap > 0 else 'higher'} mean rating for hospitals with {int(g.index[-1])}+ readmission measures <b>worse than national</b> "
             f"({g.iloc[-1]:.2f}★) than for those with none ({g.iloc[0]:.2f}★).",
             hbar([f"{int(i)}{'+' if i == g.index[-1] else ''} worse" for i in g.index], g.values,
                  [T.lerp([T.STARS[3], T.COPPER], j / max(len(g) - 1, 1)) for j in range(len(g))], ".2f"), 210)
    c1, c2 = st.columns(2)
    with c1:  # 3 hospital type
        ty = Hs.groupby("hospital_type").agg(n=(C.TARGET, "size"), m=(C.TARGET, "mean")).query("n >= 20").sort_values("m")
        card("Hospital type", f"{ty.m.iloc[-1]:.2f}★ vs {ty.m.iloc[0]:.2f}★",
             f"mean rating of <b>{ty.index[-1]}</b> vs {ty.index[0]}.",
             hbar(list(ty.index), ty.m.values, [T.STARS[2]] * len(ty), ".2f"), 170)
    with c2:  # 4 peer group
        if "cms_peer_group" in Hs:
            pg = Hs.groupby("cms_peer_group").agg(n=(C.TARGET, "size"), m=(C.TARGET, "mean"), hi=("hi", "mean")).query("n >= 20")
            spread = pg.hi.max() - pg.hi.min()
            if spread < 0.05:
                big_, txt_ = (f"{pg.hi.min():.0%}–{pg.hi.max():.0%}",
                              "share of 4–5★ whether a hospital reports 3, 4 or 5 measure groups: reporting fewer measures "
                              "does <b>not</b> go with a lower rating (CMS compares hospitals within these peer groups).")
            else:
                big_, txt_ = (f"{pg.hi.max():.0%} vs {pg.hi.min():.0%}",
                              f"share of 4–5★ for hospitals reporting {int(pg.hi.idxmax())} vs {int(pg.hi.idxmin())} measure groups.")
            card("CMS peer group", big_, txt_,
                 hbar([f"{int(i)} groups (n={int(n)})" for i, n in zip(pg.index, pg.n)], pg.hi.values, [T.PLUM] * len(pg)), 170)
    c1, c2 = st.columns(2)
    with c1:  # 5 net performance gap
        npg = Hs.groupby(C.TARGET)["mean_net_performance"].median()
        card("Engineered signal", f"{npg.iloc[-1] - npg.iloc[0]:+.2f}",
             "difference in median <b>net performance</b> between 5★ and 1★ hospitals: the single strongest driver in both "
             "permutation importance and SHAP.",
             hbar([f"{k}★" for k in npg.index], npg.values, T.STARS, "+.2f"), 200)
    with c2:  # 6 prescriptive
        if "moves_up_a_class" in Hs:
            share = Hs.moves_up_a_class.astype(float).mean()
            mv = D.get("prescriptive_moves")
            txt = "of hospitals would move up a predicted star with <b>a single change</b> in one measure result."
            fig = None
            if mv is not None:
                mv = mv.sort_values("mean_gain")
                txt += f" Biggest average lift: <b>{mv.move.iloc[-1]}</b>."
                fig = hbar([m.replace("turn one ", "").replace(" result into ", " → ") for m in mv.move], mv.mean_gain.values,
                           [T.WINE] * len(mv), "+.2f")
            card("Prescriptive", f"{share:.0%}", txt, fig, 230)
    c1, c2 = st.columns(2)
    with c1:  # 7 geography
        stt = Hs.groupby("state").agg(n=(C.TARGET, "size"), m=(C.TARGET, "mean")).query("n >= 10").sort_values("m")
        card("Geography", f"{stt.m.iloc[-1] - stt.m.iloc[0]:.2f}★",
             f"gap between the highest-rated state (<b>{stt.index[-1]}</b>, {stt.m.iloc[-1]:.2f}★) and the lowest "
             f"(<b>{stt.index[0]}</b>, {stt.m.iloc[0]:.2f}★) among states with ≥ 10 rated hospitals.",
             hbar(list(stt.index[:4]) + list(stt.index[-4:]), list(stt.m.values[:4]) + list(stt.m.values[-4:]),
                  [T.COPPER] * 4 + [T.STARS[3]] * 4, ".2f"), 230)
    with c2:  # 8 audit
        A_ = M["audit"]
        ab = Q.groupby("hospital_ownership").agg(n=("qrdi", "size"), above=("flag", lambda s: (s == "rated above profile").mean()),
                                                 below=("flag", lambda s: (s == "rated below profile").mean())).query("n >= 30")
        ab["net"] = ab.above - ab.below
        ab = ab.sort_values("net")
        card("Rating audit", f"{1 - A_['share_consistent']:.0%}",
             f"of hospitals sit a full star or more from what their own results suggest. Net share rated above profile ranges from "
             f"{ab.net.iloc[0]:+.0%} ({OWN_SHORT.get(ab.index[0], ab.index[0])}) to {ab.net.iloc[-1]:+.0%} "
             f"({OWN_SHORT.get(ab.index[-1], ab.index[-1])}).",
             hbar([OWN_SHORT.get(i, i) for i in ab.index], ab.net.values, [T.PLUM if v >= 0 else T.COPPER for v in ab.net], "+.0%"), 250)
    html('<div class="ql-caption">Findings are associations in one cross-sectional CMS release, not causal effects.</div>')

# ---------------------------------------------------------------------------
elif page == "Model performance":
    p = TM.loc[PROP]
    q_ci = ci_row("qwk", PROP)
    d_ci = q_ci if q_ci is not None and pd.notna(q_ci.get("diff_vs_ref")) else None
    hero("Evaluation on the held-out test set",
         "Does the hybrid ensemble beat the baselines?",
         f"Nine models were scored on {M['n_test']:,} unseen hospitals with every metric the brief asks for, plus "
         "ordinal metrics (quadratic-weighted κ, MAE, within-one-star) because confusing 1★ with 5★ is worse than "
         "confusing 3★ with 4★. Confidence intervals come from 1,000 paired bootstrap resamples of the test set.",
         [("Proposed: tuned RF + HGB → logistic meta-learner", "wine"), f"Best baseline: {REF}", "Tuned by QWK"])
    sig = ""
    if d_ci is not None:
        sig = (f"Δ vs {REF}: {d_ci.diff_vs_ref:+.3f} [{d_ci.diff_low:+.3f}, {d_ci.diff_high:+.3f}]"
               + (" · CI excludes 0" if d_ci.ci_excludes_zero else " · CI includes 0"))
    kpis([("Quadratic-weighted κ", f"{p.qwk:.3f}", f"95% CI {q_ci.ci_low:.3f}–{q_ci.ci_high:.3f}" if q_ci is not None else "", "wine"),
          ("Accuracy", f"{p.accuracy:.1%}", f"majority class: {TM.loc['Majority class', 'accuracy']:.1%}", ""),
          ("Macro F1", f"{p.f1_macro:.3f}", "averaged over 5 star classes", ""),
          ("ROC-AUC (OvR)", f"{p.get('roc_auc_ovr_macro', float('nan')):.3f}", "macro over classes", "plum"),
          ("Within ±1 star", f"{p.within_one_star:.1%}", f"MAE {p.mae_stars:.2f} stars", "copper")])
    if sig:
        html(f'<div class="ql-callout wine"><b>Paired comparison with the best baseline.</b> {sig}. '
             'The interval is for the difference on the same resampled hospitals, so it tests the gap directly.</div>')

    section("Model comparison with 95% bootstrap intervals")
    met = st.radio("Metric", ["qwk", "f1_macro", "accuracy"], format_func=METRIC_LABEL.get, horizontal=True, key="mc_metric")
    ci = D["bootstrap_ci"]
    d = ci[ci.metric == met].sort_values("estimate")
    colors = [T.WINE if "proposed" in m else (T.AMBER if m == REF else T.GREY) for m in d.model]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[MODEL_SHORT.get(m, m) for m in d.model], x=d.estimate, orientation="h",
                         marker=dict(color=colors, cornerradius=5),
                         error_x=dict(type="data", symmetric=False, array=d.ci_high - d.estimate, arrayminus=d.estimate - d.ci_low,
                                      color=T.INK2, thickness=1.3, width=4),
                         text=[f"{v:.3f}" for v in d.estimate], textposition="inside", insidetextanchor="start",
                         textfont=dict(color="white", size=12),
                         customdata=np.c_[d.ci_low, d.ci_high],
                         hovertemplate="%{y}<br>%{x:.3f}  [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"))
    fig.update_layout(xaxis_title=f"{METRIC_LABEL[met]} (test set)", bargap=0.3,
                      xaxis_range=[min(0, d.ci_low.min()), min(1, d.ci_high.max() + 0.05)])
    plot(fig, 70 + 42 * len(d))
    html(f'<div class="ql-caption">Wine = proposed · amber = best baseline ({REF}) · grey = other baselines.</div>')

    section("All metrics", "Macro averages treat each star class equally, so the rare 1★ and 5★ classes count as much as 3★.")
    show = TM.copy()
    show.index = [MODEL_SHORT.get(i, i) for i in show.index]
    show = show.rename(columns={"accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall",
                                "specificity_macro": "Specificity", "f1_macro": "F1 macro", "f1_weighted": "F1 wtd",
                                "qwk": "QWK", "mae_stars": "MAE (stars)", "within_one_star": "±1 star",
                                "roc_auc_ovr_macro": "ROC-AUC"})
    st.dataframe(show.style.format("{:.3f}").highlight_max(axis=0, props=f"background-color: rgba(142,42,58,.12); font-weight:600;",
                                                         subset=[c for c in show.columns if c != "MAE (stars)"])
                 .highlight_min(axis=0, props="background-color: rgba(142,42,58,.12); font-weight:600;", subset=["MAE (stars)"]),
                 **WIDE)

    c1, c2 = st.columns(2)
    with c1:
        section("Confusion matrix · stacking ensemble")
        norm = st.toggle("Show row percentages", value=True, key="cm_norm")
        cm = D["confusion_proposed"].set_index(D["confusion_proposed"].columns[0])
        z = cm.values.astype(float)
        zr = z / z.sum(axis=1, keepdims=True).clip(min=1)
        txt = [[f"{zr[i, j]:.0%}" if norm else f"{int(z[i, j])}" for j in range(5)] for i in range(5)]
        labels_ = [f"{k}★" for k in C.CLASSES]
        fig = go.Figure(go.Heatmap(z=zr, x=labels_, y=labels_,
                                   colorscale=[[i / 4, c] for i, c in enumerate(T.WINE_SEQ)], zmin=0, zmax=1, showscale=False,
                                   customdata=z, hovertemplate="official %{y} → predicted %{x}<br>%{customdata:.0f} hospitals (%{z:.0%} of row)<extra></extra>",
                                   xgap=3, ygap=3))
        for i in range(5):
            for j in range(5):
                fig.add_annotation(x=labels_[j], y=labels_[i], text=txt[i][j], showarrow=False,
                                   font=dict(size=13, color="#FFFFFF" if zr[i, j] > 0.45 else T.INK))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Official", yaxis_autorange="reversed")
        plot(fig, 420)
    with c2:
        section("One-vs-rest ROC curves", "Recomputed from the saved test-set probabilities.")
        pc = D["per_class_proposed"]
        auc = M.get("roc_auc_per_class", {})
        rc = D.get("roc_curves_proposed")
        fig = go.Figure()
        if rc is not None:
            for k, c in zip(C.CLASSES, T.STARS):
                r = rc[rc.stars == k]
                fig.add_trace(go.Scatter(x=r.fpr, y=r.tpr, mode="lines", line=dict(color=c, width=2.4),
                                         name=f"{k}★  AUC {auc.get(str(k), float('nan')):.2f}",
                                         hovertemplate=f"{k}★<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.AXIS, dash="dash", width=1), showlegend=False, hoverinfo="skip"))
        fig.update_layout(xaxis_title="False-positive rate (1 − specificity)", yaxis_title="True-positive rate (sensitivity)",
                          legend=dict(x=0.98, y=0.04, xanchor="right", yanchor="bottom", bgcolor="rgba(255,253,249,.85)"))
        plot(fig, 420 + 28)

    cal_t, cal_s = D.get("calibration"), M.get("calibration", {})
    if cal_t is not None and cal_s:
        section("Can the model's confidence be trusted?",
                "Reliability diagram: hospitals grouped by the predicted probability of the most likely class; a calibrated "
                "model sits on the diagonal.")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.AXIS, dash="dash", width=1),
                                     name="perfect calibration", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cal_t.mean_confidence, y=cal_t.accuracy, mode="lines+markers", name="stacking ensemble",
                                     line=dict(color=T.WINE, width=2.4), marker=dict(size=9, color=T.WINE,
                                                                                      line=dict(color=T.CARD, width=2)),
                                     customdata=cal_t.n, hovertemplate="confidence %{x:.2f} → accuracy %{y:.2f}<br>n = %{customdata}<extra></extra>"))
            fig.update_layout(xaxis=dict(title="Predicted probability of most likely class", range=[0, 1]),
                              yaxis=dict(title="Observed accuracy", range=[0, 1.02]), legend=dict(x=0.02, y=0.98))
            plot(fig, 360)
        with c2:
            gap = cal_s["mean_confidence"] - cal_s["accuracy"]
            kpis([("Expected calibration error", f"{cal_s['ece']:.3f}", "0 = perfectly calibrated", "wine"),
                  ("Multi-class Brier score", f"{cal_s['brier_multiclass']:.3f}", "lower is better (0–2)", "plum"),
                  ("Confidence vs accuracy", f"{gap:+.3f}", f"{'over' if gap > 0 else 'under'}-confident on average", "copper")])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("Per-class report · stacking ensemble")
        pcs = pc.copy()
        pcs["stars"] = pcs["stars"].map(lambda k: f"{k}★")
        st.dataframe(pcs.style.format({c: "{:.3f}" for c in pcs.columns if c not in ("stars", "support")}), hide_index=True, **WIDE)
        cv = M.get("cv", {})
        if cv:
            section("5-fold cross-validation (whole dataset)")
            table(pd.DataFrame([{"Model": MODEL_SHORT.get(k, k), "QWK": f"{v['qwk_mean']:.3f} ± {v['qwk_std']:.3f}",
                                 "Macro F1": f"{v['f1_macro_mean']:.3f} ± {v['f1_macro_std']:.3f}"} for k, v in cv.items()]))
    with c2:
        section("Ablation · what each feature family adds", "Same random forest, different inputs.")
        ab = D["ablation"].set_index(D["ablation"].columns[0]).sort_values("qwk")
        fig = go.Figure(go.Bar(y=ab.index, x=ab.qwk, orientation="h",
                               marker=dict(color=[T.WINE if i == ab.qwk.idxmax() else T.GREY for i in ab.index], cornerradius=5),
                               text=[f"{v:.3f}" for v in ab.qwk], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: QWK %{x:.3f}<extra></extra>"))
        fig.update_layout(xaxis=dict(title="Quadratic-weighted κ", range=[0, min(1.05, ab.qwk.max() + 0.12)]), bargap=0.38)
        plot(fig, 330)

# ---------------------------------------------------------------------------
elif page == "Explainability":
    hero("Model-agnostic and model-specific explanations", "What actually drives the predicted rating?",
         "Permutation importance shuffles one input at a time on the test set and measures the drop in κ for the full "
         "stacking model. SHAP decomposes each random-forest prediction into feature contributions. Agreement between the "
         "two is the evidence that a driver is real rather than an artefact of one method.",
         [("Permutation importance · stacking", "wine"), "TreeSHAP · tuned random forest"])
    top = st.slider("Features shown", 5, 25, 12, key="ex_top")
    perm = D["permutation_importance"].head(top)
    c1, c2 = st.columns(2)
    with c1:
        section("Permutation importance", "Drop in quadratic-weighted κ when the feature is shuffled (mean ± sd).")
        plot(importance_chart([pretty(f) for f in perm.feature], perm.importance_mean.values, perm.importance_std.values,
                              T.WINE, "Drop in κ"), 90 + 30 * len(perm))
    sh = D.get("shap_importance")
    with c2:
        if sh is not None:
            sh = sh.rename(columns={sh.columns[0]: "feature"}).head(top)
            section("Mean |SHAP| value", "Average absolute contribution to the class probabilities.")
            plot(importance_chart([pretty(f) for f in sh.feature], sh.mean_abs_shap.values, None, T.PLUM, "Mean |SHAP|"),
                 90 + 30 * len(sh))
        else:
            st.info("SHAP was not available in this run.")
    if sh is not None:
        section("Which star class each feature speaks to", "Mean |SHAP| per class; darker = larger contribution.")
        cls_cols = [c for c in sh.columns if c.startswith("stars_")]
        z = sh[cls_cols].values
        fig = go.Figure(go.Heatmap(z=z, x=[c.replace("stars_", "") + "★" for c in cls_cols], y=[pretty(f) for f in sh.feature],
                                   colorscale=[[i / 4, c] for i, c in enumerate(T.WINE_SEQ)], xgap=3, ygap=3,
                                   colorbar=dict(thickness=10, outlinewidth=0, title="|SHAP|"),
                                   hovertemplate="%{y} → %{x}: %{z:.4f}<extra></extra>"))
        fig.update_layout(yaxis_autorange="reversed")
        plot(fig, 90 + 26 * len(sh))
        tp = set(D["permutation_importance"].feature.head(5))
        ts = list(D["shap_importance"].iloc[:, 0].head(8))
        overlap = [f for f in ts if any(f == p or f.startswith(p) for p in tp)]
        html(f'<div class="ql-callout">Top-5 permutation features also in the SHAP top-8: <b>{len(overlap)}</b> '
             f'({", ".join(pretty(f) for f in overlap) or "none"}). SHAP works on one-hot / missing-indicator columns, '
             'so categorical features are split across several rows.</div>')

# ---------------------------------------------------------------------------
elif page == "Rating audit (QRDI)":
    A = M["audit"]
    hero("Quality Rating Discrepancy Index", "Is every hospital's star in line with its own profile?",
         "A random forest that sees <b>only</b> measure-performance and coverage features (no ownership, type or state) "
         "predicts each hospital's star distribution out-of-fold. <b>QRDI = official stars − expected stars.</b> "
         "Positive: rated above what similar profiles usually receive; negative: below.",
         [("Out-of-fold, 5 folds", "wine"), f"Flag threshold |QRDI| ≥ {A['threshold']:g}", "Kruskal–Wallis + ε²"])
    kpis([("Consistent", f"{A['share_consistent']:.1%}", f"|QRDI| < {A['threshold']:g}", ""),
          ("Rated above profile", f"{A['share_above']:.1%}", "official ≥ expected + 1", "plum"),
          ("Rated below profile", f"{A['share_below']:.1%}", "official ≤ expected − 1", "copper"),
          ("Spearman ρ", f"{A['spearman_official_vs_expected']:.3f}", "official vs expected", "wine")])
    html('<div class="ql-callout wine"><b>Read with care.</b> CMS computes the stars from continuous measure scores that '
         'this file does not contain, using the same measure groups. QRDI is therefore an internal-consistency screen: it '
         'points to hospitals worth a closer look, it does not show that any hospital was mis-rated.</div>')
    c1, c2 = st.columns(2)
    with c1:
        section("Distribution of QRDI")
        bins = np.linspace(Q.qrdi.min() - 0.05, Q.qrdi.max() + 0.05, 46)
        cnt, edges = np.histogram(Q.qrdi, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        lim = 2.0
        fig = go.Figure(go.Bar(x=mids, y=cnt, width=np.diff(edges) * 0.92,
                               marker=dict(color=[T.lerp([T.COPPER, T.MID, T.PLUM], (np.clip(m, -lim, lim) + lim) / (2 * lim)) for m in mids]),
                               hovertemplate="QRDI %{x:+.2f}: %{y} hospitals<extra></extra>"))
        for t_ in (-A["threshold"], A["threshold"]):
            fig.add_vline(x=t_, line=dict(color=T.INK2, dash="dash", width=1))
        fig.add_vline(x=0, line=dict(color=T.INK, width=1))
        fig.add_annotation(x=0.99, xref="paper", y=cnt.max(), text="rated above →", showarrow=False, xanchor="right", font=dict(color=T.PLUM))
        fig.add_annotation(x=0.01, xref="paper", y=cnt.max(), text="← rated below", showarrow=False, xanchor="left", font=dict(color=T.COPPER))
        fig.update_layout(xaxis_title="QRDI = official − expected", yaxis_title="Hospitals", bargap=0)
        plot(fig, 380)
    with c2:
        section("Expected vs official rating", "Boxes should climb the dotted diagonal.")
        plot(box_by_star(Q.expected_stars, Q.official_stars, "Expected stars (out-of-fold)", diag=True), 380)

    section("Does QRDI differ systematically between groups?")
    groups = sorted(k.replace("qrdi_by_", "") for k in D if k.startswith("qrdi_by_"))
    nice = {"hospital_ownership": "Ownership", "hospital_type": "Hospital type", "state": "State",
            "emergency_services": "Emergency services", "birthing_friendly": "Birthing-friendly"}
    g = st.selectbox("Group by", groups, format_func=lambda x: nice.get(x, x), key="audit_group",
                     index=groups.index("hospital_ownership") if "hospital_ownership" in groups else 0)
    t = D[f"qrdi_by_{g}"].sort_values("mean")
    gt = M["audit_group_tests"].get(g, {})
    if "p_value" in gt:
        eps = gt["epsilon_squared"]
        size = "negligible" if eps < 0.01 else "small" if eps < 0.06 else "moderate" if eps < 0.14 else "large"
        html(f'<div class="ql-callout">Kruskal–Wallis H = <b>{gt["H"]:.2f}</b>, p = <b>{gt["p_value"]:.3g}</b>, '
             f'ε² = <b>{eps:.3f}</b> ({size} effect) across {gt["groups_tested"]} groups with ≥ {C.MIN_GROUP_N} hospitals. '
             'With thousands of hospitals even a tiny difference can reach p &lt; 0.05, so read the effect size first.</div>')
    labels = [OWN_SHORT.get(x, x) + f"  (n={n})" for x, n in zip(t[g], t["n"])]
    colors = [T.PLUM if m >= 0 else T.COPPER for m in t["mean"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["mean"], y=labels, mode="markers", marker=dict(size=11, color=colors, line=dict(color=T.CARD, width=2)),
                             error_x=dict(type="data", symmetric=False, array=t.ci_high - t["mean"], arrayminus=t["mean"] - t.ci_low,
                                          color=T.INK2, thickness=1.3, width=0),
                             customdata=np.c_[t.share_above, t.share_below],
                             hovertemplate="%{y}<br>mean QRDI %{x:+.3f}<br>above: %{customdata[0]:.1%} · below: %{customdata[1]:.1%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=T.INK, width=1))
    fig.update_layout(xaxis_title="Mean QRDI with 95% CI   ← below profile · above profile →", showlegend=False)
    plot(fig, 90 + 30 * len(t))

    if "state" in Q:
        section("Mean QRDI by state")
        n_state = Q.groupby("state").size()
        plot(tile_map(Q.groupby("state")["qrdi"].mean(), n_state, diverging=True, fmt="{:+.2f}", label="Mean QRDI"), 380)

    section("Largest discrepancies", "Top hospitals in each direction. A starting point for review, not a verdict.")
    o = D["qrdi_top_outliers"].copy()
    o["direction"] = np.where(o.qrdi > 0, "rated above profile", "rated below profile")
    o = o.rename(columns={C.NAME_COL: "hospital", "hospital_ownership": "ownership", "official_stars": "official",
                          "expected_stars": "expected"})
    table(o[[c for c in ["hospital", "state", "ownership", "official", "expected", "qrdi", "direction"] if c in o]].round(2))

# ---------------------------------------------------------------------------
elif page == "Quality archetypes":
    sg = M.get("segmentation")
    if not sg or H is None or "archetype" not in H:
        st.info("This results folder has no archetype analysis. Re-run `python -m qualitylens run`.")
        st.stop()
    k = int(sg["k"])
    hero("Unsupervised segmentation · k-means + PCA", "Hospital quality archetypes",
         "Hospitals are grouped only by their measure-performance profile (net performance, worse share, coverage); the star "
         "rating is <b>not</b> used to form the groups. Archetypes are then numbered by their mean official rating, A1 lowest, "
         "and drawn with the same gold→bronze ramp as the stars.",
         [(f"k = {k} archetypes", "wine"), "Silhouette + Davies–Bouldin", "PCA map", "Kruskal–Wallis on stars"])
    kpis([("Archetypes", f"{k}", "best silhouette for k ≥ 3", "wine"),
          ("Silhouette", f"{sg['silhouette']:.3f}", "−1 … 1, higher = better separated", ""),
          ("Davies–Bouldin", f"{sg['davies_bouldin']:.3f}", "lower = better separated", "plum"),
          ("PCA variance (2 PCs)", f"{sum(sg['pca_explained_variance']):.0%}", "of the profile shown on the map", ""),
          ("Stars across archetypes", f"ε² {sg['stars_epsilon_squared']:.3f}", f"Kruskal–Wallis p = {sg['stars_kruskal_p']:.2g}", "copper")])
    sel = D["cluster_selection"]
    c1, c2 = st.columns(2)
    for col_, met, lab in [(c1, "silhouette", "Silhouette (higher is better)"), (c2, "davies_bouldin", "Davies–Bouldin (lower is better)")]:
        with col_:
            fig = go.Figure(go.Scatter(x=sel.k, y=sel[met], mode="lines+markers", line=dict(color=T.WINE, width=2.4),
                                       marker=dict(size=[16 if kk == k else 8 for kk in sel.k],
                                                   color=[T.STARS[1] if kk == k else T.WINE for kk in sel.k],
                                                   line=dict(color=T.CARD, width=2)),
                                       hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
            fig.update_layout(title=lab, xaxis_title="number of clusters k", xaxis=dict(dtick=1))
            plot(fig, 260)
    ap = D["archetype_profiles"]
    section("Map of all rated hospitals", "First two principal components of the standardised profile. Hover for the hospital.")
    fig = go.Figure()
    for a in range(1, k + 1):
        d_ = H[H.archetype == a]
        fig.add_trace(go.Scattergl(x=d_.pc1, y=d_.pc2, mode="markers", name=f"A{a}  (n={len(d_)})",
                                   marker=dict(size=6, color=arche_color(a, k), opacity=0.78, line=dict(width=0)),
                                   text=d_[C.NAME_COL].str.title(), customdata=np.c_[d_[C.TARGET], d_.state],
                                   hovertemplate="%{text} (%{customdata[1]})<br>official %{customdata[0]}★<extra>A" + str(a) + "</extra>"))
    ev = sg["pca_explained_variance"]
    fig.update_layout(xaxis_title=f"PC1 ({ev[0]:.0%} of variance)", yaxis_title=f"PC2 ({ev[1]:.0%} of variance)",
                      legend=dict(orientation="h", y=1.08, x=0, itemsizing="constant"))
    plot(fig, 470)

    c1, c2 = st.columns([1.25, 1])
    with c1:
        section("What defines each archetype", "Mean z-score of each profile feature vs all hospitals (plum = above, copper = below).")
        z = D["archetype_zscores"].set_index(D["archetype_zscores"].columns[0])
        lim = float(np.abs(z.values).max())
        fig = go.Figure(go.Heatmap(z=z.values, x=[pretty(c) for c in z.columns], y=[f"A{a}" for a in z.index],
                                   colorscale=[[0, T.COPPER], [0.5, T.MID], [1, T.PLUM]], zmin=-lim, zmax=lim, xgap=3, ygap=3,
                                   colorbar=dict(thickness=10, outlinewidth=0, title="z"),
                                   hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>"))
        fig.update_layout(xaxis=dict(tickangle=-40), yaxis=dict(autorange="reversed"))
        plot(fig, 220 + 50 * k)
    with c2:
        section("Star mix by archetype", "Share of each official rating inside the archetype.")
        fig = go.Figure()
        for kk, c in zip(C.CLASSES, T.STARS):
            fig.add_trace(go.Bar(y=[f"A{a}" for a in ap.archetype], x=ap[f"share_{kk}_star"], orientation="h", name=f"{kk}★",
                                 marker=dict(color=c, line=dict(color=T.CARD, width=2)),
                                 hovertemplate=f"%{{y}} · {kk}★: %{{x:.0%}}<extra></extra>"))
        fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=1.12, x=0, traceorder="normal"))
        plot(fig, 220 + 50 * k)
    section("Archetype summary")
    show_ = ap[["archetype", "name", "n", "mean_stars"] + [f"share_{kk}_star" for kk in (4, 5)]].copy()
    show_["share 4–5★"] = show_.share_4_star + show_.share_5_star
    table(show_.drop(columns=["share_4_star", "share_5_star"]).rename(columns={"name": "description", "mean_stars": "mean ★"})
          .round(3))
    html('<div class="ql-callout">Archetypes are descriptive groupings of similar profiles. They are useful for '
         'benchmarking a hospital against peers that look like it, not for ranking hospitals.</div>')

# ---------------------------------------------------------------------------
elif page == "Hospital explorer":
    hero("Look up any rated hospital", "Hospital profile & audit card",
         "Search by name, filter by state or audit flag, and see how a hospital's published measure results compare with "
         "the national picture, what star rating its profile would typically receive, and how far the official rating sits from that.")
    if H is None:
        st.warning("This results folder has no hospital_profiles table. Re-run the pipeline to create it.")
        st.stop()
    c1, c2, c3 = st.columns([1, 1, 2])
    states = ["All"] + sorted(H["state"].dropna().unique())
    s_sel = c1.selectbox("State", states, key="ex_state")
    f_sel = c2.selectbox("Audit flag", ["All", "consistent", "rated above profile", "rated below profile"], key="ex_flag")
    v = H if s_sel == "All" else H[H.state == s_sel]
    v = v if f_sel == "All" else v[v.flag == f_sel]
    city = "citytown" if "citytown" in v else None
    v = v.sort_values(C.NAME_COL)
    opts = v.index.tolist()
    if not opts:
        st.info("No hospitals match these filters.")
        st.stop()
    fmt = (lambda i: f"{H.at[i, C.NAME_COL]} · {H.at[i, city].title() if city and isinstance(H.at[i, city], str) else ''}, {H.at[i, 'state']}")
    pick = c3.selectbox(f"Hospital ({len(opts):,} matching)", opts, format_func=fmt, key="ex_pick")
    r = H.loc[pick]

    left, right = st.columns([1, 1.4])
    with left:
        loc = f"{str(r.get('citytown', '')).title()}, {r['state']}" if city else r["state"]
        html(f'<div class="ql-card"><div class="ql-eyebrow">{r["hospital_type"]} · {OWN_SHORT.get(r["hospital_ownership"], r["hospital_ownership"])}</div>'
             f'<h4 style="font-size:1.35rem;margin-top:.3rem">{str(r[C.NAME_COL]).title()}</h4><div class="ql-caption">{loc} · CMS ID {r[C.ID_COL]}</div>'
             f'<div style="display:flex;gap:1.6rem;margin:.8rem 0 .4rem;flex-wrap:wrap">'
             f'<div><div class="ql-caption" style="margin:0">Official rating</div>{stars_html(r[C.TARGET])}</div>'
             f'<div><div class="ql-caption" style="margin:0">Expected from profile</div><span style="font-family:Fraunces,serif;font-size:1.4rem">{r.expected_stars:.2f}★</span></div>'
             f'<div><div class="ql-caption" style="margin:0">QRDI</div><span style="font-family:Fraunces,serif;font-size:1.4rem">{r.qrdi:+.2f}</span></div></div>'
             f'{flag_badge(r.flag)}'
             f'<div class="ql-caption" style="margin-top:.8rem">Emergency services: {r.get("emergency_services", "—")} · '
             f'Birthing-friendly: {r.get("birthing_friendly", "—")} · Measure groups with outcome data: {int(r.get("domains_with_data", 0))}</div></div>')
        p = [r[f"p_{k}"] for k in C.CLASSES]
        fig = go.Figure(go.Bar(x=[f"{k}★" for k in C.CLASSES], y=p,
                               marker=dict(color=[T.WINE if k == r[C.TARGET] else T.lerp([T.CARD, c], .75) for k, c in zip(C.CLASSES, T.STARS)],
                                           cornerradius=5),
                               text=[f"{x:.0%}" for x in p], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_layout(title="Profile-based probability · wine = official", yaxis=dict(tickformat=".0%", range=[0, max(p) * 1.3 + 0.02]))
        plot(fig, 300)
    with right:
        section("Where this hospital sits among all rated hospitals",
                "Net performance = (better − worse) / reported. Light band: middle 80% of hospitals; dark band: middle 50%; "
                "tick: national median; wine dot: this hospital.")
        fig = go.Figure()
        lab, xs_lo, xs_hi = [], [], []
        for d in C.OUTCOME_DOMAINS:
            col = f"{d}_net_performance_ratio"
            if col not in H:
                continue
            s_ = H[col].dropna()
            p10, p25, p50, p75, p90 = s_.quantile([.1, .25, .5, .75, .9])
            name = DOMAIN_LABELS[d]
            lab.append(name)
            fig.add_trace(go.Scatter(x=[p10, p90], y=[name, name], mode="lines", line=dict(color=T.MID, width=16),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[p25, p75], y=[name, name], mode="lines", line=dict(color=T.GREY, width=16),
                                     hovertemplate=f"{name}<br>middle 50%: {p25:+.2f} to {p75:+.2f}<extra></extra>", showlegend=False))
            fig.add_trace(go.Scatter(x=[p50], y=[name], mode="markers", marker=dict(symbol="line-ns", size=22, line=dict(color=T.INK2, width=2)),
                                     hovertemplate=f"national median {p50:+.2f}<extra></extra>", showlegend=False))
            v_ = r[col]
            if pd.notna(v_):
                fig.add_trace(go.Scatter(x=[v_], y=[name], mode="markers+text", text=[f"{v_:+.2f}"], textposition="top center",
                                         textfont=dict(color=T.WINE, size=12), cliponaxis=False,
                                         marker=dict(size=15, color=T.WINE, line=dict(color=T.CARD, width=2)),
                                         hovertemplate=f"this hospital {v_:+.2f}<extra></extra>", showlegend=False))
                xs_lo.append(v_); xs_hi.append(v_)
            xs_lo.append(p10); xs_hi.append(p90)
        if lab:
            fig.add_vline(x=0, line=dict(color=T.AXIS, width=1, dash="dot"))
            fig.update_layout(xaxis=dict(title="Net performance ratio", range=[min(xs_lo) - 0.12, max(xs_hi) + 0.12], zeroline=False),
                              yaxis=dict(autorange="reversed"))
            plot(fig, 260)
        cov = []
        for d in C.DOMAINS:
            col = f"{d}_coverage"
            if col in H:
                cov.append((DOMAIN_LABELS[d], r[col], H[col].median()))
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[c[0] for c in cov], x=[c[2] for c in cov], orientation="h", name="National median", marker=dict(color=T.GREY, cornerradius=4)))
        fig.add_trace(go.Bar(y=[c[0] for c in cov], x=[0 if pd.isna(c[1]) else c[1] for c in cov], orientation="h", name="This hospital",
                             marker=dict(color=T.AMBER, cornerradius=4)))
        fig.update_layout(barmode="group", xaxis=dict(title="Coverage (share of group measures reported)", tickformat=".0%", range=[0, 1.05]),
                          legend=dict(orientation="h", y=1.12, x=0), bargap=0.3)
        plot(fig, 300)
        bw = []
        for d in C.OUTCOME_DOMAINS:
            b, w, n = (f"count_of_{d}_measures_{k}" for k in ("better", "worse", "no_different"))
            if b in H:
                bw.append({"Measure group": DOMAIN_LABELS[d], "Better than national": r[b], "No different": r[n],
                           "Worse than national": r[w], "Reported": r[f"count_of_facility_{d}_measures"]})
        table(pd.DataFrame(bw).fillna("—"))

    bundle = safe_bundle()
    rowd = r.to_dict()
    t1, t2, t3 = st.tabs(["Why this rating?", "Path to the next star", "Peer twins"])
    with t1:
        if bundle is None:
            st.info("Needs results/model.joblib (run the pipeline once).")
        else:
            typical = {c: H[c].median() for cols in GROUPS.values() for c in cols
                       if c in H and pd.api.types.is_numeric_dtype(H[c])}
            typical.update({c: H[c].mode().iat[0] for c in C.CONTEXT_CATEGORICAL if c in H})
            le = local_explanation(bundle, rowd, typical)
            exp_now = float(PR.score(bundle["model"], [rowd], bundle["num_cols"], bundle["cat_cols"])[0][0])
            html(f'<div class="ql-caption">Each bar: how much the model\'s expected rating ({exp_now:.2f}★) changes because this '
                 'hospital differs from a typical hospital in that group of inputs (occlusion analysis on the stacking model). '
                 'Plum lifts the rating, copper pulls it down. (The card above uses the separate performance-only '
                 'audit model, so its expected rating can differ.)</div>')
            fig = go.Figure(go.Bar(y=le.group, x=le.contribution, orientation="h",
                                   marker=dict(color=[T.PLUM if v >= 0 else T.COPPER for v in le.contribution], cornerradius=4),
                                   text=[f"{v:+.2f}★" for v in le.contribution], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x:+.3f} stars<extra></extra>"))
            lim = max(0.15, float(le.contribution.abs().max()) * 1.35)
            fig.add_vline(x=0, line=dict(color=T.INK, width=1))
            fig.update_layout(xaxis=dict(title="Effect on expected stars vs a typical hospital", range=[-lim, lim]), bargap=0.3)
            plot(fig, 380)
    with t2:
        if bundle is None:
            st.info("Needs results/model.joblib (run the pipeline once).")
        else:
            path = PR.path_to_next_star(bundle["model"], rowd, bundle["num_cols"], bundle["cat_cols"])
            start, end = path.iloc[0], path.iloc[-1]
            if len(path) == 1:
                html('<div class="ql-callout">No single-measure improvement raises this hospital\'s expected rating '
                     '(it may already be predicted 5★, or report too few outcome measures).</div>')
            else:
                moved = end.most_likely > start.most_likely
                html(f'<div class="ql-callout wine"><b>{len(path) - 1} change{"s" if len(path) > 2 else ""}</b> '
                     f'{"move" if moved else "would not yet move"} the most likely rating from {int(start.most_likely)}★ '
                     f'{"to " + str(int(end.most_likely)) + "★" if moved else ""}; expected stars '
                     f'{start.expected_stars:.2f} → {end.expected_stars:.2f}. Model-based what-if, not a causal claim.</div>')
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    fig = go.Figure(go.Scatter(x=path.step, y=path.expected_stars, mode="lines+markers+text",
                                               line=dict(color=T.WINE, width=2.6), text=[f"{v:.2f}" for v in path.expected_stars],
                                               textposition="top center", cliponaxis=False,
                                               marker=dict(size=12, color=[T.lerp(T.STARS, (m - 1) / 4) for m in path.most_likely],
                                                           line=dict(color=T.CARD, width=2)),
                                               customdata=path.change, hovertemplate="step %{x}: %{customdata}<extra></extra>"))
                    for kk in range(1, 6):
                        fig.add_hline(y=kk + 0.5, line=dict(color=T.LINE, width=1, dash="dot"))
                    fig.update_layout(xaxis=dict(title="step", dtick=1), yaxis=dict(title="expected stars",
                                      range=[max(1, path.expected_stars.min() - 0.4), min(5.2, path.expected_stars.max() + 0.4)]))
                    plot(fig, 300)
                with c2:
                    table(path.rename(columns={"expected_stars": "expected ★", "most_likely": "most likely ★"}).round(2))
    with t3:
        nn, Z = peer_index(folder)
        pos = H.index.get_loc(pick)
        _, ind = nn.kneighbors(Z[pos:pos + 1])
        peers = H.iloc[[i for i in ind[0] if i != pos][:5]]
        pm = peers[C.TARGET].mean()
        kpis([("This hospital", f"{int(r[C.TARGET])}★", f"expected {r.expected_stars:.2f}★ from profile", "wine"),
              ("Its 5 closest peers", f"{pm:.1f}★", "mean official rating", "plum"),
              ("Difference", f"{r[C.TARGET] - pm:+.1f}★", "official minus peers' mean", "copper")])
        pcols = [c for c in [C.NAME_COL, "citytown", "state", "hospital_type", C.TARGET, "expected_stars", "archetype"] if c in peers]
        pt = peers[pcols].rename(columns={C.NAME_COL: "Peer hospital", "citytown": "City", "state": "State",
                                          "hospital_type": "Type", C.TARGET: "Official ★", "expected_stars": "Expected ★",
                                          "archetype": "Archetype"})
        pt["Peer hospital"] = pt["Peer hospital"].str.title()
        if "City" in pt:
            pt["City"] = pt["City"].astype(str).str.title()
        table(pt.round(2))
        html('<div class="ql-caption">Peers = nearest neighbours on the standardised measure-performance profile '
             '(the same features used for the archetypes).</div>')

    section("Matching hospitals", "Sorted by the size of the discrepancy.")
    cols = [c for c in [C.NAME_COL, "citytown", "state", "hospital_type", "hospital_ownership", C.TARGET, "expected_stars", "qrdi", "flag"] if c in v]
    vv = v[cols].rename(columns={C.NAME_COL: "Hospital", "citytown": "City", "state": "State", "hospital_type": "Type",
                                 "hospital_ownership": "Ownership", C.TARGET: "Official ★", "expected_stars": "Expected ★",
                                 "qrdi": "QRDI", "flag": "Audit flag"})
    if "Ownership" in vv:
        vv["Ownership"] = vv["Ownership"].map(lambda x: OWN_SHORT.get(x, x))
    if "City" in vv:
        vv["City"] = vv["City"].astype(str).str.title()
    table(vv.reindex(vv.QRDI.abs().sort_values(ascending=False).index).head(300).round(2))

# ---------------------------------------------------------------------------
elif page == "What-if simulator":
    if not (ROOT / folder / "model.joblib").exists():
        st.warning("The trained model (model.joblib) is not in this results folder. Run `python -m qualitylens run` once "
                   "to create it; the other pages work without it.")
        st.stop()
    try:
        bundle = load_bundle(folder)
    except Exception as exc:  # e.g. pickled with a different scikit-learn version
        st.warning(f"Could not load model.joblib ({type(exc).__name__}). Re-run `python -m qualitylens run` with the "
                   "installed library versions to recreate it.")
        st.stop()
    d, ch = bundle["defaults"], bundle["choices"]
    hero("Interactive prediction with the saved stacking model", "What would move a hospital's rating?",
         "Set the measure results for a hypothetical hospital, or start from a real one, and see the full probability "
         "distribution over 1–5 stars. Engineered features are recomputed with the same code used in training.")
    base = dict(d)
    for dom in C.DOMAINS:  # measures defined per group (constant across hospitals, so not a model input)
        k = f"{dom}_group_measure_count"
        if pd.isna(base.get(k, np.nan)) and H is not None and k in H:
            base[k] = H[k].max()
    if H is not None:
        start = st.selectbox("Start from", ["National median profile"] + H.sort_values(C.NAME_COL).index.tolist(),
                             format_func=lambda i: i if isinstance(i, str) else f"{H.at[i, C.NAME_COL]} ({H.at[i, 'state']}) · official {H.at[i, C.TARGET]}★",
                             key="wi_start")
        if not isinstance(start, str):
            hr = H.loc[start]
            for k in list(base):
                if k in hr and pd.notna(hr[k]):
                    base[k] = hr[k]
    key_sfx = f"_{start}" if H is not None else ""
    row = dict(base)

    def ival(k, fallback=0):
        x = base.get(k, fallback)
        return int(x) if pd.notna(x) else fallback

    cols = st.columns(3)
    for col, dom in zip(cols, C.OUTCOME_DOMAINS):
        with col.container(border=True):
            html(f'<div class="ql-eyebrow" style="margin-bottom:.2rem">Outcome group</div>'
                 f'<div style="font-family:Fraunces,serif;font-size:1.15rem;font-weight:600;color:{T.INK}">{DOMAIN_LABELS[dom]}</div>')
            tot = max(ival(f"{dom}_group_measure_count", 1), 1)
            rep = st.slider("Measures reported", 0, max(tot, 1), min(ival(f"count_of_facility_{dom}_measures"), tot), key=f"r{dom}{key_sfx}")
            better = st.slider("Better than national", 0, max(rep, 1), min(ival(f"count_of_{dom}_measures_better"), rep), key=f"b{dom}{key_sfx}")
            worse = st.slider("Worse than national", 0, max(rep, 1), min(ival(f"count_of_{dom}_measures_worse"), rep), key=f"w{dom}{key_sfx}")
            if better + worse > rep:
                st.warning("better + worse exceeds measures reported")
            row.update({f"count_of_facility_{dom}_measures": rep, f"{dom}_group_measure_count": max(tot, rep),
                        f"count_of_{dom}_measures_better": better, f"count_of_{dom}_measures_worse": worse,
                        f"count_of_{dom}_measures_no_different": max(rep - better - worse, 0)})
    cols = st.columns(2)
    for col, dom in zip(cols, ["pt_exp", "te"]):
        with col:
            tot = max(ival(f"{dom}_group_measure_count", 1), 1)
            rep = st.slider(f"{DOMAIN_LABELS[dom]}: measures reported", 0, tot, min(ival(f"count_of_facility_{dom}_measures"), tot), key=f"r{dom}{key_sfx}")
            row.update({f"count_of_facility_{dom}_measures": rep, f"{dom}_group_measure_count": tot})
    ctx = st.columns(len(bundle["cat_cols"]))
    for col, name in zip(ctx, bundle["cat_cols"]):
        opts = ch[name]
        cur = base.get(name, d[name])
        row[name] = col.selectbox(name.replace("_", " ").capitalize(), opts, index=opts.index(cur) if cur in opts else 0, key=f"c{name}{key_sfx}")

    X = F.build_features(pd.DataFrame([row])).reindex(columns=bundle["num_cols"] + bundle["cat_cols"])
    proba = bundle["model"].predict_proba(X)[0]
    k_best = C.CLASSES[int(np.argmax(proba))]
    expv = float(proba @ np.array(C.CLASSES))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Most likely rating", f"{k_best}★")
        html(f'<div class="ql-card"><div class="ql-caption" style="margin:0">Expected rating Σ k·P(k)</div>'
             f'<div style="font-family:Fraunces,serif;font-size:1.6rem;color:{T.INK}">{expv:.2f}★</div>'
             f'<div class="ql-caption" style="margin:.5rem 0 0">Ordinal decoding (rounded expectation)</div>'
             f'{stars_html(int(np.clip(round(expv), 1, 5)))}</div>')
    with c2:
        fig = go.Figure(go.Bar(x=[f"{k}★" for k in C.CLASSES], y=proba,
                               marker=dict(color=[T.WINE if k == k_best else c for k, c in zip(C.CLASSES, T.STARS)], cornerradius=6),
                               text=[f"{x:.0%}" for x in proba], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_layout(title="Predicted probability of each star rating", yaxis=dict(tickformat=".0%", range=[0, max(proba) * 1.3]))
        plot(fig, 320)

# ---------------------------------------------------------------------------
else:
    hero("Reproducible pipeline", "Method, data and provenance",
         "Everything on this dashboard is produced by one command, <code>python -m qualitylens run</code>, from the public CMS file. "
         "Preprocessing lives inside scikit-learn pipelines so nothing from the test set leaks into training.")
    steps = [("01", "Acquire", "CMS API, paged; SHA-256 provenance"), ("02", "Clean", "'Not Available' → missing; unrated rows dropped; consistency checks"),
             ("03", "Engineer", "net-performance ratios, worse share, coverage"), ("04", "Model", "5 baselines + tuned RF/HGB stacking"),
             ("05", "Evaluate", "brief metrics + κ, MAE; paired bootstrap; 5-fold CV"), ("06", "Explain", "permutation importance + SHAP"),
             ("07", "Audit", "out-of-fold QRDI; Kruskal–Wallis, ε²"), ("08", "Segment", "k-means archetypes; silhouette, DB; PCA"),
             ("09", "Prescribe", "greedy what-if: path to the next star")]
    html('<div class="ql-flow">' + "".join(f'<div class="ql-step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d_}</div></div>'
                                            for n, t, d_ in steps) + "</div>")
    c1, c2 = st.columns(2)
    with c1:
        section("Dataset")
        table(pd.DataFrame({"Item": ["Name", "Publisher", "Dataset page", "HealthData.gov listing", "Retrieved (UTC)", "Raw rows × columns",
                                     "Rated hospitals (modelled)", "Target", "SHA-256 of raw file"],
                            "Value": [C.DATASET_NAME, "Centers for Medicare & Medicaid Services", C.DATASET_PAGE, C.MIRROR_PAGE,
                                      str(PROV.get("retrieved_utc", "—")), f"{CL['rows_raw']:,} × {M.get('dataset', {}).get('raw_columns', '—')}",
                                      f"{CL['rows_clean']:,}", f"{C.TARGET} (1–5, ordinal)", str(PROV.get("sha256", "—"))[:24] + "…"]}))
        section("Cleaning report")
        cc = CL.get("consistency_checks", {})
        table(pd.DataFrame({"Step": ["'Not Available' / blank tokens converted to missing", "Duplicate facility IDs dropped",
                                     "Rows without a star rating (excluded, target not imputed)", "Parts > reported (any outcome group)",
                                     "Reported > group total (any group)", "Columns with negative counts",
                                     "Zero-variance columns removed before modelling"],
                            "Result": [f"{CL.get('missing_tokens_converted', 0):,}", CL.get("duplicates_dropped", 0), f"{CL.get('rows_without_rating', 0):,}",
                                       sum(v for k, v in cc.items() if k.endswith("exceed_reported")),
                                       sum(v for k, v in cc.items() if k.endswith("exceeds_total")),
                                       len(cc.get("columns_with_negative_counts", [])),
                                       len(CL.get("constant_features_dropped", []))]}).astype(str))
    with c2:
        section("What is novel")
        html('<div class="ql-card"><h4>1 · Engineered net-performance features</h4>(better − worse) / reported per outcome group, '
             'plus worse-share and coverage, with an ablation that measures what they add.</div>'
             '<div class="ql-card"><h4>2 · Ordinal-aware hybrid ensemble</h4>Tuned random forest + gradient boosting blended by a '
             'logistic meta-learner, tuned on quadratic-weighted κ, with ordinal decoding and paired-bootstrap comparison against 5 baselines.</div>'
             '<div class="ql-card"><h4>3 · QRDI consistency audit</h4>Out-of-fold expected rating from performance-only features; '
             'group differences tested with Kruskal–Wallis and ε² effect sizes.</div>'
             '<div class="ql-card"><h4>4 · Archetypes + prescriptive what-if</h4>Unsupervised quality archetypes (k-means, '
             'silhouette / Davies–Bouldin, PCA map) and a per-hospital "path to the next star" built on the trained model.</div>')
        section("Tools & versions")
        vers = INFO.get("versions", {})
        table(pd.DataFrame({"Library": list(vers), "Version": [str(x) for x in vers.values()]}))

html(f'<div class="ql-foot">QualityLens · data: CMS Provider Data Catalog, {C.DATASET_NAME} ({C.DATASET_ID}) · '
     'Star ratings are a CMS summary of many measures; this tool explains and screens them, it does not re-rate hospitals.</div>')
