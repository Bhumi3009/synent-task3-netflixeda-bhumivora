import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix EDA Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
  }

  /* App background */
  .stApp {
    background: linear-gradient(160deg, #0d0d0d 0%, #141414 60%, #1a0a0a 100%);
  }

  /* Hero header */
  .hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.06em;
    color: #E50914;
    line-height: 1;
    margin-bottom: 0;
    text-shadow: 0 0 60px rgba(229,9,20,0.4);
  }
  .hero-subtitle {
    font-size: 1rem;
    color: #888;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(229,9,20,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: rgba(229,9,20,0.6); }
  .metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    color: #E50914;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-top: 4px;
  }

  /* Section headings */
  .section-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.7rem;
    color: #fff;
    letter-spacing: 0.08em;
    border-left: 4px solid #E50914;
    padding-left: 12px;
    margin: 28px 0 16px;
  }

  /* Insight cards */
  .insight-card {
    background: rgba(229,9,20,0.07);
    border-left: 3px solid #E50914;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #d4d4d4;
  }
  .insight-num {
    font-family: 'Bebas Neue', sans-serif;
    color: #E50914;
    font-size: 1.1rem;
    margin-right: 6px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid rgba(229,9,20,0.2);
  }
  [data-testid="stSidebar"] .sidebar-content { padding: 1rem; }

  /* Plotly chart containers */
  .stPlotlyChart { border-radius: 12px; overflow: hidden; }

  /* Divider */
  hr { border-color: rgba(229,9,20,0.15) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #111; }
  ::-webkit-scrollbar-thumb { background: #E50914; border-radius: 3px; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Selectbox & multiselect */
  .stSelectbox [data-baseweb="select"] > div,
  .stMultiSelect [data-baseweb="select"] > div {
    background: #1c1c1c !important;
    border-color: rgba(229,9,20,0.3) !important;
    color: #e8e8e8 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────────
NETFLIX_RED   = "#E50914"
NETFLIX_DARK  = "#141414"
ACCENT_WARM   = "#FF6B35"
ACCENT_COOL   = "#5BC0EB"
COLOR_SEQ     = [NETFLIX_RED, "#ff4d4d", ACCENT_WARM, "#FFD166", ACCENT_COOL, "#06D6A0",
                 "#9B5DE5", "#F15BB5", "#FEE440", "#00BBF9"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#ccc"),
    title_font=dict(family="Bebas Neue", size=20, color="#fff"),
    margin=dict(t=50, b=40, l=40, r=20),
    hoverlabel=dict(bgcolor="#1c1c1c", bordercolor="#E50914", font_color="#fff"),
    colorway=COLOR_SEQ,
)

# ─── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_added"]  = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"]  = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month_name()
    return df

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:#E50914;
                letter-spacing:0.1em;margin-bottom:4px;">🎬 NETFLIX EDA</div>
    <div style="font-size:0.7rem;color:#666;letter-spacing:0.2em;
                text-transform:uppercase;margin-bottom:20px;">Dashboard Controls</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload netflix_titles.csv", type=["csv"])
    st.markdown("---")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw["date_added"]  = pd.to_datetime(df_raw["date_added"], errors="coerce")
        df_raw["year_added"]  = df_raw["date_added"].dt.year
        df_raw["month_added"] = df_raw["date_added"].dt.month_name()
        df = df_raw.copy()
        st.success(f"✔ Loaded {len(df):,} rows")
    else:
        st.info("Upload your dataset to get started.\n\nExpecting: **netflix_titles.csv**")
        st.stop()

    st.markdown("#### Filters")
    content_type = st.multiselect(
        "Content Type",
        options=df["type"].dropna().unique().tolist(),
        default=df["type"].dropna().unique().tolist(),
    )

    year_min = int(df["year_added"].min()) if not df["year_added"].isna().all() else 2008
    year_max = int(df["year_added"].max()) if not df["year_added"].isna().all() else 2021
    year_range = st.slider("Year Added", year_min, year_max, (year_min, year_max))

    top_n = st.slider("Top N (Countries / Genres)", 5, 20, 10)

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#555;text-align:center'>Built with Streamlit · Netflix Dataset</div>", unsafe_allow_html=True)

# ─── FILTER DATA ────────────────────────────────────────────────────────────────
df_f = df[
    df["type"].isin(content_type) &
    df["year_added"].between(year_range[0], year_range[1], inclusive="both")
]

# ─── HERO ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 8px">
  <div class="hero-title">Netflix Content</div>
  <div class="hero-subtitle">Exploratory Data Analysis Dashboard</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── KPI METRICS ────────────────────────────────────────────────────────────────
movies_count  = len(df_f[df_f["type"] == "Movie"])
shows_count   = len(df_f[df_f["type"] == "TV Show"])
countries_cnt = df_f["country"].nunique()
genres_cnt    = df_f["listed_in"].str.split(", ").explode().nunique()

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [f"{len(df_f):,}", f"{movies_count:,}", f"{shows_count:,}", f"{countries_cnt:,}"],
    ["Total Titles", "Movies", "TV Shows", "Countries"]
):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ROW 1: Type Split + Ratings ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">Content Breakdown</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 2])

with col_a:
    type_counts = df_f["type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig_pie = go.Figure(go.Pie(
        labels=type_counts["type"],
        values=type_counts["count"],
        hole=0.55,
        marker=dict(colors=[NETFLIX_RED, ACCENT_COOL], line=dict(color="#141414", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,} titles<extra></extra>",
    ))
    fig_pie.update_layout(
        **LAYOUT_BASE,
        title="Movies vs TV Shows",
        showlegend=False,
        height=300,
        annotations=[dict(text=f"<b>{len(df_f):,}</b>", x=0.5, y=0.5,
                          font=dict(size=22, color="#fff", family="Bebas Neue"),
                          showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    ratings = df_f["rating"].value_counts().reset_index()
    ratings.columns = ["rating", "count"]
    fig_rating = go.Figure(go.Bar(
        x=ratings["count"],
        y=ratings["rating"],
        orientation="h",
        marker=dict(
            color=ratings["count"],
            colorscale=[[0, "#330000"], [1, NETFLIX_RED]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
    ))
    fig_rating.update_layout(
        **LAYOUT_BASE,
        title="Content Ratings Distribution",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
        height=300,
    )
    st.plotly_chart(fig_rating, use_container_width=True)

# ─── ROW 2: Content Over Time ────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Growth Over Time</div>', unsafe_allow_html=True)

yearly = df_f.groupby(["year_added", "type"]).size().reset_index(name="count")
fig_line = px.area(
    yearly, x="year_added", y="count", color="type",
    color_discrete_map={"Movie": NETFLIX_RED, "TV Show": ACCENT_COOL},
    labels={"year_added": "Year", "count": "Titles Added", "type": ""},
)
fig_line.update_traces(line=dict(width=2.5))
fig_line.update_layout(
    **LAYOUT_BASE,
    title="Titles Added to Netflix Per Year",
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=""),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=""),
    height=320,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_line, use_container_width=True)

# ─── ROW 3: Countries + Genres ───────────────────────────────────────────────────
st.markdown('<div class="section-heading">Geography & Genres</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    top_countries = (
        df_f["country"]
        .str.split(", ").explode()
        .str.strip().dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_countries.columns = ["country", "count"]
    fig_country = px.bar(
        top_countries.sort_values("count"),
        x="count", y="country", orientation="h",
        color="count",
        color_continuous_scale=["#330000", NETFLIX_RED, ACCENT_WARM],
    )
    fig_country.update_layout(
        **LAYOUT_BASE,
        title=f"Top {top_n} Countries by Content Volume",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        yaxis=dict(title=""),
        height=400,
    )
    st.plotly_chart(fig_country, use_container_width=True)

with col_d:
    top_genres = (
        df_f["listed_in"]
        .str.split(", ").explode()
        .str.strip().dropna()
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_genres.columns = ["genre", "count"]
    fig_genre = px.bar(
        top_genres.sort_values("count"),
        x="count", y="genre", orientation="h",
        color="count",
        color_continuous_scale=["#001a33", ACCENT_COOL, "#00ffe0"],
    )
    fig_genre.update_layout(
        **LAYOUT_BASE,
        title=f"Top {top_n} Genres on Netflix",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        yaxis=dict(title=""),
        height=400,
    )
    st.plotly_chart(fig_genre, use_container_width=True)

# ─── ROW 4: Release Year + Heatmap ───────────────────────────────────────────────
st.markdown('<div class="section-heading">Release Year & Correlation</div>', unsafe_allow_html=True)
col_e, col_f = st.columns([2, 1])

with col_e:
    fig_hist = px.histogram(
        df_f.dropna(subset=["release_year"]),
        x="release_year", nbins=40,
        color_discrete_sequence=[NETFLIX_RED],
    )
    fig_hist.update_traces(marker_line_width=0, opacity=0.85)
    fig_hist.update_layout(
        **LAYOUT_BASE,
        title="Distribution of Release Years",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"),
        height=300,
        bargap=0.04,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_f:
    numeric_df = df_f.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#003080"], [0.5, "#141414"], [1, NETFLIX_RED]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
            showscale=False,
        ))
        fig_heat.update_layout(
            **LAYOUT_BASE, title="Correlation Heatmap",
            height=300,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# ─── INSIGHTS ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">Key Insights</div>', unsafe_allow_html=True)

insights = [
    "Movies outnumber TV Shows — the catalog skews toward standalone films.",
    "Netflix content additions surged dramatically after 2015, peaking around 2019–2020.",
    "The United States dominates content production, followed by India and the UK.",
    "TV-MA is the most frequent rating, reflecting a mature audience focus.",
    "Drama and International Movies are the top genres, indicating a global-first strategy.",
]

cols_ins = st.columns(2)
for i, insight in enumerate(insights):
    with cols_ins[i % 2]:
        st.markdown(f"""
        <div class="insight-card">
          <span class="insight-num">0{i+1}</span>{insight}
        </div>
        """, unsafe_allow_html=True)

# ─── RAW DATA ────────────────────────────────────────────────────────────────────
with st.expander("🗂 View Raw Data"):
    st.dataframe(
        df_f.head(200),
        use_container_width=True,
        height=320,
    )