import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

st.set_page_config(page_title="VECTOR.FC", layout="wide", page_icon="⚡")

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL   = "#4cc6dd"
BG     = "#0c0f14"
CARD   = "#141820"
BORDER = "#1e2530"
MUTED  = "#8899aa"
TEXT   = "#e8eaed"
GREEN  = "#2ecc71"
RED    = "#e74c3c"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Base ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
.block-container {{ padding: 0 2.5rem 3rem 2.5rem !important; max-width: 100% !important; }}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
}}
[data-testid="stSelectbox"] label {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}}

/* ── Metric ── */
[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}}

/* ── Nav buttons ── */
button[data-testid="baseButton-primary"] {{
    background-color: {TEAL} !important;
    color: {BG} !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 5px !important;
}}
button[data-testid="baseButton-secondary"] {{
    background-color: transparent !important;
    color: {MUTED} !important;
    border: 1px solid {BORDER} !important;
    font-weight: 500 !important;
    border-radius: 5px !important;
}}
button[data-testid="baseButton-secondary"]:hover {{
    border-color: {TEAL} !important;
    color: {TEAL} !important;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 1rem 0 !important; }}

/* ── Scrollbars ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

footer, #MainMenu {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'pass_completion_pct', 'progressive_passes_p90', 'pct_passes_forward',
    'pct_passes_sideways', 'pct_passes_backward', 'avg_pass_distance',
    'key_passes_p90', 'crosses_p90', 'long_balls_p90', 'pressures_p90',
    'pressure_success_rate', 'tackles_p90', 'interceptions_p90',
    'aerial_duels_p90', 'progressive_carries_p90', 'dribble_success_rate',
    'carries_final_third_p90', 'xg_p90', 'xg_per_shot', 'shots_p90',
    'box_touches_p90',
]

FEATURE_LABELS = {
    'pass_completion_pct':     'Pass Completion %',
    'progressive_passes_p90':  'Progressive Passes',
    'pct_passes_forward':      'Forward Pass %',
    'pct_passes_sideways':     'Sideways Pass %',
    'pct_passes_backward':     'Backward Pass %',
    'avg_pass_distance':       'Avg Pass Distance',
    'key_passes_p90':          'Key Passes',
    'crosses_p90':             'Crosses',
    'long_balls_p90':          'Long Balls',
    'pressures_p90':           'Pressures',
    'pressure_success_rate':   'Pressure Success Rate',
    'tackles_p90':             'Tackles',
    'interceptions_p90':       'Interceptions',
    'aerial_duels_p90':        'Aerial Duels',
    'progressive_carries_p90': 'Progressive Carries',
    'dribble_success_rate':    'Dribble Success Rate',
    'carries_final_third_p90': 'Carries into Final Third',
    'xg_p90':                  'xG',
    'xg_per_shot':             'xG per Shot',
    'shots_p90':               'Shots',
    'box_touches_p90':         'Box Touches',
}

OVERLAP_LABELS = [
    (0.92, "Elite overlap"),
    (0.85, "Very high overlap"),
    (0.75, "High overlap"),
    (0.60, "Moderate overlap"),
    (0.00, "Low overlap"),
]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    profiles = pd.read_csv("data/player_profiles_clustered.csv")
    raw      = pd.read_csv("data/features_raw.csv")

    # Raw pass completion (grouped across all seasons for each player)
    grp = raw.groupby("player_id")[["passes_completed", "passes_attempted"]].sum()
    grp["raw_pass_pct"] = grp["passes_completed"] / grp["passes_attempted"] * 100
    profiles = profiles.merge(grp[["raw_pass_pct"]], on="player_id", how="left")

    # Approximate appearances
    profiles["approx_apps"] = (profiles["minutes_played"] / 90).round().clip(lower=1).astype(int)

    # Display label for selectbox
    profiles["display_name"] = profiles["player_name"] + "  ·  " + profiles["team_name"]

    # Scaled matrix → L2-norm (for cosine similarity)
    X        = profiles[FEATURE_COLS].values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    norms    = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1, norms)
    X_norm   = X_scaled / norms

    # Percentile rank per feature (0–100), ranked on raw profile values
    pct = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        ranks    = rankdata(X[:, j], method="average")
        pct[:, j] = (ranks - 1) / (len(ranks) - 1) * 100

    return profiles, X_norm, pct


profiles, X_norm, PERCENTILES = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
def player_index(display):
    return profiles.index[profiles["display_name"] == display][0]


def get_similar_players(idx, n=10):
    scores       = X_norm @ X_norm[idx]
    scores[idx]  = -1
    top          = np.argsort(scores)[::-1][:n]
    rows         = profiles.iloc[top][["player_name", "team_name", "cluster_label"]].copy()
    rows["sim"]  = scores[top]
    rows.index   = range(1, n + 1)
    return rows


def get_contributions(idx_a, idx_b):
    contribs = X_norm[idx_a] * X_norm[idx_b]
    cos_sim  = float(contribs.sum())
    df = pd.DataFrame({
        "feature":      FEATURE_COLS,
        "label":        [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "contribution": contribs,
        "val_a":        profiles.iloc[idx_a][FEATURE_COLS].values,
        "val_b":        profiles.iloc[idx_b][FEATURE_COLS].values,
    }).sort_values("contribution", ascending=False).reset_index(drop=True)
    return df, cos_sim


def overlap_label(score):
    for threshold, label in OVERLAP_LABELS:
        if score >= threshold:
            return label
    return "Low overlap"


# ── Reusable HTML blocks ──────────────────────────────────────────────────────
def badge(text, color=TEAL):
    return (
        f'<span style="background:{color}22; color:{color}; border:1px solid {color}55; '
        f'border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:600; '
        f'letter-spacing:0.06em;">{text}</span>'
    )


def perc_bar(label, perc, teal=TEAL):
    perc = min(max(perc, 0), 100)
    ordinal = (
        f"{int(perc)}{'th' if 4 <= int(perc) % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(int(perc)%10,'th')}"
    )
    return f"""
<div style="margin:0 0 10px 0;">
  <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
    <span style="color:{MUTED}; font-size:0.75rem; letter-spacing:0.04em;">{label}</span>
    <span style="color:{teal}; font-size:0.75rem; font-weight:700;">{ordinal}</span>
  </div>
  <div style="height:4px; background:{BORDER}; border-radius:2px; overflow:hidden;">
    <div style="height:100%; width:{perc:.1f}%; background:{teal}; border-radius:2px;"></div>
  </div>
</div>"""


def sim_row(rank, name, team, cluster, sim):
    bar_w = int(sim * 100)
    return f"""
<div style="display:flex; align-items:center; padding:9px 0; border-bottom:1px solid {BORDER};">
  <span style="color:{TEAL}; font-size:0.7rem; font-weight:700; width:22px; flex-shrink:0;">{rank:02d}</span>
  <div style="flex:1; min-width:0; padding-right:12px;">
    <div style="color:{TEXT}; font-size:0.85rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{name}</div>
    <div style="color:{MUTED}; font-size:0.72rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{team} &nbsp;·&nbsp; {cluster}</div>
  </div>
  <div style="text-align:right; flex-shrink:0;">
    <div style="color:{TEAL}; font-size:0.9rem; font-weight:700; margin-bottom:4px;">{sim:.2f}</div>
    <div style="width:60px; height:3px; background:{BORDER}; border-radius:2px; overflow:hidden; margin-left:auto;">
      <div style="height:100%; width:{bar_w}%; background:{TEAL};"></div>
    </div>
  </div>
</div>"""


def contrib_bar(label, val_a, val_b, contribution, name_a, name_b, positive=True):
    color     = GREEN if positive else RED
    max_abs   = 0.35
    bar_w     = min(abs(contribution) / max_abs * 100, 100)
    sign      = "+" if contribution >= 0 else ""
    return f"""
<div style="margin-bottom:18px;">
  <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px;">
    <span style="color:{TEXT}; font-size:0.82rem;">{label}</span>
    <span style="color:{color}; font-size:0.82rem; font-weight:700;">{sign}{contribution:.3f}</span>
  </div>
  <div style="height:4px; background:{BORDER}; border-radius:2px; overflow:hidden; margin-bottom:6px;">
    <div style="height:100%; width:{bar_w:.1f}%; background:{color}; border-radius:2px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between;">
    <span style="color:{MUTED}; font-size:0.68rem;">{name_a}: {val_a:.2f}</span>
    <span style="color:{MUTED}; font-size:0.68rem;">{name_b}: {val_b:.2f}</span>
  </div>
</div>"""


# ── Navigation header ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Profile"

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:18px 0 20px 0; border-bottom:1px solid {BORDER}; margin-bottom:28px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="width:28px; height:28px; background:{TEAL}; border-radius:5px;
                display:flex; align-items:center; justify-content:center;
                font-size:0.8rem; color:{BG}; font-weight:900;">V</div>
    <span style="color:{TEXT}; font-size:1rem; font-weight:800; letter-spacing:0.12em;">VECTOR.FC</span>
  </div>
  <div style="color:{MUTED}; font-size:0.72rem; letter-spacing:0.08em;">
    DATASET &nbsp;·&nbsp; {len(profiles):,} PLAYERS
  </div>
</div>
""", unsafe_allow_html=True)

nav_left, nav_mid, nav_right = st.columns([3, 2, 3])
with nav_mid:
    btn_left, btn_right = st.columns(2)
    with btn_left:
        if st.button(
            "Profile",
            use_container_width=True,
            type="primary" if st.session_state.page == "Profile" else "secondary",
        ):
            st.session_state.page = "Profile"
            st.rerun()
    with btn_right:
        if st.button(
            "Compare",
            use_container_width=True,
            type="primary" if st.session_state.page == "Compare" else "secondary",
        ):
            st.session_state.page = "Compare"
            st.rerun()

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── Page 1 — Profile ──────────────────────────────────────────────────────────
if st.session_state.page == "Profile":

    display_names = sorted(profiles["display_name"].tolist())
    left, right   = st.columns([4, 6], gap="large")

    with left:
        choice = st.selectbox("Search player", display_names, label_visibility="visible")
        idx    = player_index(choice)
        row    = profiles.iloc[idx]

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # Top-8 features by percentile rank
        player_pcts = PERCENTILES[idx]
        feature_perc_pairs = sorted(
            zip(FEATURE_COLS, player_pcts), key=lambda x: x[1], reverse=True
        )[:8]

        bars_html = ""
        for feat, perc in feature_perc_pairs:
            bars_html += perc_bar(FEATURE_LABELS[feat], perc)

        st.markdown(bars_html, unsafe_allow_html=True)

    with right:
        # Player name + cluster badge
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<span style="color:{TEXT}; font-size:1.5rem; font-weight:800;">{row["player_name"]}</span>'
            f'</div>'
            f'<div style="margin-bottom:20px; display:flex; align-items:center; gap:10px;">'
            f'<span style="color:{MUTED}; font-size:0.82rem;">{row["team_name"]}</span>'
            f'&nbsp;{badge(row["cluster_label"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Headline stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Apps",    f'{row["approx_apps"]:,}')
        m2.metric("Minutes", f'{int(row["minutes_played"]):,}')
        m3.metric("Pass %",  f'{row["raw_pass_pct"]:.1f}')
        m4.metric("xG / 90", f'{row["xg_p90"]:.2f}')

        st.markdown(f"<hr style='border-color:{BORDER};margin:18px 0;'>", unsafe_allow_html=True)

        # Similar players header
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:baseline; '
            f'margin-bottom:4px;">'
            f'<span style="color:{TEXT}; font-size:0.85rem; font-weight:700; '
            f'letter-spacing:0.06em;">MOST SIMILAR PLAYERS</span>'
            f'<span style="color:{MUTED}; font-size:0.7rem;">COSINE SIM</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        similar   = get_similar_players(idx)
        rows_html = ""
        for rank, r in similar.iterrows():
            rows_html += sim_row(rank, r["player_name"], r["team_name"], r["cluster_label"], r["sim"])

        st.markdown(
            f'<div style="overflow-y:auto; max-height:420px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )


# ── Page 2 — Compare ─────────────────────────────────────────────────────────
else:
    display_names = sorted(profiles["display_name"].tolist())

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        choice_a = st.selectbox("Player A", display_names, key="pa", label_visibility="visible")
    with col_vs:
        st.markdown(
            f'<div style="height:68px; display:flex; align-items:flex-end; '
            f'justify-content:center; color:{MUTED}; font-size:0.8rem; '
            f'font-weight:700; letter-spacing:0.1em;">VS</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        default_b = display_names[1] if display_names[0] == choice_a else display_names[0]
        choice_b  = st.selectbox("Player B", display_names, index=display_names.index(default_b), key="pb", label_visibility="visible")

    idx_a = player_index(choice_a)
    idx_b = player_index(choice_b)
    row_a = profiles.iloc[idx_a]
    row_b = profiles.iloc[idx_b]

    df_contrib, cos_sim = get_contributions(idx_a, idx_b)

    # ── Player cards + similarity score ──────────────────────────────────────
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    card_a, score_col, card_b = st.columns([4, 3, 4])

    with card_a:
        st.markdown(
            f'<div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px; '
            f'padding:20px; text-align:center;">'
            f'<div style="color:{TEXT}; font-size:1.2rem; font-weight:800; margin-bottom:4px;">'
            f'{row_a["player_name"]}</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem; margin-bottom:12px;">'
            f'{row_a["team_name"]}</div>'
            f'{badge(row_a["cluster_label"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with score_col:
        overlap = overlap_label(cos_sim)
        st.markdown(
            f'<div style="text-align:center; padding:10px 0;">'
            f'<div style="color:{MUTED}; font-size:0.65rem; letter-spacing:0.18em; '
            f'text-transform:uppercase; margin-bottom:6px;">COSINE SIMILARITY</div>'
            f'<div style="color:{TEAL}; font-size:4rem; font-weight:900; line-height:1.05;">'
            f'{cos_sim:.2f}</div>'
            f'<div style="color:{MUTED}; font-size:0.78rem; margin-top:6px;">{overlap}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with card_b:
        st.markdown(
            f'<div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px; '
            f'padding:20px; text-align:center;">'
            f'<div style="color:{TEXT}; font-size:1.2rem; font-weight:800; margin-bottom:4px;">'
            f'{row_b["player_name"]}</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem; margin-bottom:12px;">'
            f'{row_b["team_name"]}</div>'
            f'{badge(row_b["cluster_label"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"<hr style='border-color:{BORDER};margin:24px 0 20px 0;'>", unsafe_allow_html=True)

    # ── Breakdown columns ─────────────────────────────────────────────────────
    name_a = row_a["player_name"].split()[-1]
    name_b = row_b["player_name"].split()[-1]

    sim_col, diff_col = st.columns(2, gap="large")

    with sim_col:
        st.markdown(
            f'<div style="color:{GREEN}; font-size:0.75rem; font-weight:700; '
            f'letter-spacing:0.1em; margin-bottom:16px;">● STRONGEST SIMILARITIES</div>',
            unsafe_allow_html=True,
        )
        top5 = df_contrib.head(5)
        for _, r in top5.iterrows():
            st.markdown(
                contrib_bar(r["label"], r["val_a"], r["val_b"], r["contribution"],
                            name_a, name_b, positive=True),
                unsafe_allow_html=True,
            )

    with diff_col:
        st.markdown(
            f'<div style="color:{RED}; font-size:0.75rem; font-weight:700; '
            f'letter-spacing:0.1em; margin-bottom:16px;">● BIGGEST DIFFERENTIATORS</div>',
            unsafe_allow_html=True,
        )
        bot5 = df_contrib.tail(5).sort_values("contribution")
        for _, r in bot5.iterrows():
            st.markdown(
                contrib_bar(r["label"], r["val_a"], r["val_b"], r["contribution"],
                            name_a, name_b, positive=False),
                unsafe_allow_html=True,
            )
