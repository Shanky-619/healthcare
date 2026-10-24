"""Visual identity for the QualityLens dashboard: colour tokens, CSS and a Plotly template.

Palette (validated for colour-blind separation and contrast on the ivory surface):
  * star rating 1..5  -> gold -> bronze ordinal ramp
  * proposed model    -> wine; baselines -> warm grey
  * QRDI sign         -> copper (rated below profile) / plum (rated above profile), warm-grey midpoint
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

INK, INK2, MUTED = "#1F1A17", "#4E4640", "#8A8079"
PAGE, SURFACE, CARD, LINE, AXIS = "#F6F1EA", "#FBF8F3", "#FFFDF9", "#E8E1D6", "#C9BFB2"
WINE, WINE_DARK, WINE_DEEP = "#8E2A3A", "#5A1724", "#3A1219"
AMBER, PLUM, COPPER, GREY = "#C27A1A", "#5B3F8C", "#B5452B", "#B9B0A5"
MID = "#ECE6DC"
STARS = ["#D4A03C", "#BB812A", "#9A6420", "#774A18", "#553311"]
GOLD_LIGHT = "#F3E2BF"
WINE_SEQ = [SURFACE, "#EBC9C2", "#C27377", WINE, WINE_DARK]


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp(stops: list[str], t: float) -> str:
    """Colour at position t (0..1) along a list of hex stops."""
    t = min(max(float(t), 0.0), 1.0)
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    f = seg - i
    a, b = _hex(stops[i]), _hex(stops[i + 1])
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def is_dark(hex_color: str) -> bool:
    r, g, b = (c / 255 for c in _hex(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5


FONT_BODY = "Inter, 'Segoe UI', system-ui, sans-serif"
FONT_DISPLAY = "Fraunces, Georgia, 'Times New Roman', serif"

pio.templates["qualitylens"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT_BODY, color=INK2, size=13),
        title=dict(font=dict(family=FONT_BODY, color=INK, size=15), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[WINE, AMBER, PLUM],
        xaxis=dict(gridcolor=LINE, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, ticks="", title_font=dict(color=INK2)),
        yaxis=dict(gridcolor=LINE, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, ticks="", title_font=dict(color=INK2)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(family=FONT_BODY, color=INK, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        margin=dict(l=18, r=22, t=44, b=18),
        bargap=0.28,
    )
)
pio.templates.default = "qualitylens"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT_BODY}; }}
.stApp {{ background: {PAGE}; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ font-family: {FONT_DISPLAY} !important; color: {INK} !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.55rem !important; }}
h3 {{ font-size: 1.2rem !important; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(175deg, {WINE_DEEP} 0%, {WINE_DARK} 55%, #4A1520 100%);
  border-right: none;
}}
section[data-testid="stSidebar"] * {{ color: #F4E9DD !important; }}
section[data-testid="stSidebar"] .ql-brand {{ font-family: {FONT_DISPLAY}; font-size: 1.85rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }}
section[data-testid="stSidebar"] .ql-brand span {{ color: {STARS[0]} !important; }}
section[data-testid="stSidebar"] .ql-tag {{ font-size: .78rem; opacity: .72; margin-top: .35rem; letter-spacing: .02em; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(244,233,221,.16); margin: 1rem 0; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .5rem .7rem; border-radius: 10px; margin-bottom: 2px; transition: background .15s ease; width: 100%;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.07); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: rgba(212,160,60,.18); box-shadow: inset 3px 0 0 {STARS[0]}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none !important; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{ background: rgba(255,255,255,.08) !important; border-color: rgba(255,255,255,.22) !important; }}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {{ color: #F4E9DD !important; -webkit-text-fill-color: #F4E9DD !important; background: transparent !important; }}
section[data-testid="stSidebar"] code {{ background: rgba(255,255,255,.12); color: #FFF6EA !important; padding: 0 .3rem; border-radius: 4px; }}
.ql-side-card {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 12px; padding: .8rem .9rem; font-size: .8rem; line-height: 1.55; }}
.ql-side-card b {{ color: #FFF6EA !important; }}

/* ---------- hero ---------- */
.ql-hero {{
  background: radial-gradient(1200px 300px at 100% 0%, rgba(212,160,60,.18), transparent 60%),
              linear-gradient(120deg, {CARD} 0%, {SURFACE} 100%);
  border: 1px solid {LINE}; border-radius: 20px; padding: 1.6rem 1.9rem 1.4rem; margin-bottom: 1.2rem;
  box-shadow: 0 1px 2px rgba(31,26,23,.04), 0 10px 30px -18px rgba(90,23,36,.25);
}}
.ql-eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: {WINE}; }}
.ql-title {{ font-family: {FONT_DISPLAY}; font-size: 2.1rem; font-weight: 600; color: {INK}; line-height: 1.15; margin: .35rem 0 .45rem; }}
.ql-sub {{ color: {INK2}; font-size: .98rem; max-width: 860px; line-height: 1.55; }}
.ql-chips {{ margin-top: .9rem; display: flex; gap: .45rem; flex-wrap: wrap; }}
.ql-chip {{ font-size: .75rem; padding: .28rem .65rem; border-radius: 999px; background: {MID}; color: {INK2}; border: 1px solid {LINE}; }}
.ql-chip.wine {{ background: rgba(142,42,58,.09); color: {WINE}; border-color: rgba(142,42,58,.2); }}

/* ---------- KPI cards ---------- */
.ql-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem; margin: .2rem 0 1.2rem; }}
.ql-kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 1rem 1.1rem .9rem; position: relative; overflow: hidden; }}
.ql-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: {STARS[0]}; }}
.ql-kpi.wine::before {{ background: {WINE}; }}
.ql-kpi.plum::before {{ background: {PLUM}; }}
.ql-kpi.copper::before {{ background: {COPPER}; }}
.ql-kpi .lbl {{ font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {MUTED}; }}
.ql-kpi .val {{ font-family: {FONT_DISPLAY}; font-size: 1.95rem; font-weight: 600; color: {INK}; line-height: 1.15; margin-top: .25rem; }}
.ql-kpi .note {{ font-size: .8rem; color: {INK2}; margin-top: .2rem; }}

/* ---------- cards / callouts ---------- */
.ql-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 1.1rem 1.25rem; margin-bottom: .9rem; }}
.ql-card h4 {{ font-family: {FONT_DISPLAY}; margin: 0 0 .4rem; font-size: 1.08rem; color: {INK}; }}
.ql-callout {{ background: rgba(212,160,60,.10); border: 1px solid rgba(187,129,42,.28); border-left: 4px solid {STARS[1]};
  border-radius: 12px; padding: .85rem 1.05rem; color: {INK2}; font-size: .92rem; line-height: 1.55; margin: .3rem 0 1rem; }}
.ql-callout.wine {{ background: rgba(142,42,58,.06); border-color: rgba(142,42,58,.22); border-left-color: {WINE}; }}
.ql-section {{ font-family: {FONT_DISPLAY}; font-size: 1.25rem; font-weight: 600; color: {INK}; margin: 1.3rem 0 .15rem; }}
.ql-caption {{ color: {MUTED}; font-size: .84rem; margin-bottom: .5rem; }}
.ql-badge {{ display: inline-block; font-size: .74rem; font-weight: 600; padding: .22rem .6rem; border-radius: 999px; }}
.ql-badge.above {{ background: rgba(91,63,140,.12); color: {PLUM}; }}
.ql-badge.below {{ background: rgba(181,69,43,.12); color: {COPPER}; }}
.ql-badge.ok {{ background: {MID}; color: {INK2}; }}
.ql-stars {{ color: {STARS[2]}; letter-spacing: .08em; font-size: 1.25rem; }}
.ql-stars .off {{ color: {LINE}; }}
.ql-flow {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch; margin: .4rem 0 1rem; }}
.ql-step {{ flex: 1 1 130px; background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: .75rem .8rem; position: relative; }}
.ql-step .n {{ font-size: .7rem; font-weight: 700; color: {WINE}; letter-spacing: .1em; }}
.ql-step .t {{ font-weight: 600; color: {INK}; font-size: .92rem; margin: .15rem 0 .2rem; }}
.ql-step .d {{ font-size: .78rem; color: {INK2}; line-height: 1.4; }}
.stMarkdown code, .ql-hero code {{ color: {WINE}; background: rgba(142,42,58,.07); border-radius: 5px; padding: .05rem .35rem; font-size: .85em; }}
.ql-foot {{ color: {MUTED}; font-size: .78rem; margin-top: 2.5rem; border-top: 1px solid {LINE}; padding-top: .8rem; }}

/* ---------- streamlit widgets ---------- */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][class*="border"] {{ border-radius: 16px !important; }}
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-color: {LINE} !important; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: .9rem 1.1rem; }}
[data-testid="stMetricValue"] {{ font-family: {FONT_DISPLAY}; color: {INK}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem .9rem; border-radius: 10px 10px 0 0; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {WINE} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {WINE}; }}
div[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 0; overflow: hidden; box-sizing: border-box; }}
.stAlert {{ border-radius: 12px; }}
</style>
"""
