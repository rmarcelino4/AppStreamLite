"""
FPF Video Tagger · Dashboard de Análise — Sessão UEFA Pro
=========================================================
streamlit run fpf_dashboard.py

Contexto: 5 grupos analisam a 1ª parte de 5 jogos diferentes de Portugal.
          Os dados são depois agregados para detectar padrões e tendências.

Requisitos: pip install streamlit pandas matplotlib seaborn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── LOGIN ───────────────────────────────────────────────────────────────────
AUTH_USER = "UEFAPRO"
AUTH_PASS = "Quiaios2026"

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("Acesso Restrito")
    st.write("Insere as credenciais para continuar.")
    user = st.text_input("Utilizador", value="", key="auth_user")
    pwd = st.text_input("Password", value="", type="password", key="auth_pass")
    if st.button("Entrar"):
        if user == AUTH_USER and pwd == AUTH_PASS:
            st.session_state.auth_ok = True
            st.success("Autenticado com sucesso.")
            st.rerun()
        else:
            st.error("Credenciais inválidas.")
    st.stop()

# ── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FPF · Análise UEFA Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMA / ESTILO ─────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "BG": "#0E1120",
        "SIDEBAR": "#080A12",
        "CARD": "#141828",
        "BORDER": "#1E2540",
        "TEXT": "#C8D6EE",
        "MUTED": "#4A5A7A",
        "GOLD": "#E8B84B",
        "GREEN": "#3BDB6A",
        "BLUE": "#5B9EF5",
        "RED": "#E84855",
        "CYAN": "#22D3EE",
        "ORANGE": "#FF8C42",
    },
    "light": {
        "BG": "#F6F7FB",
        "SIDEBAR": "#FFFFFF",
        "CARD": "#FFFFFF",
        "BORDER": "#E3E7F0",
        "TEXT": "#1B2430",
        "MUTED": "#5B6B7A",
        "GOLD": "#B8860B",
        "GREEN": "#1E9D57",
        "BLUE": "#2563EB",
        "RED": "#DC2626",
        "CYAN": "#0891B2",
        "ORANGE": "#F97316",
    },
}

style_choice = st.sidebar.selectbox(
    "Estilo de visualização",
    ["Escuro", "Claro"],
    index=1,
    key="style_choice",
)

T = THEMES["dark" if style_choice == "Escuro" else "light"]



def apply_css():
    st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {T['BG']}; color: {T['TEXT']}; }}
  [data-testid="stSidebar"]          {{ background: {T['SIDEBAR']}; border-right: 1px solid {T['BORDER']}; }}
  [data-testid="metric-container"]   {{ background: {T['CARD']}; border: 1px solid {T['BORDER']};
                                        border-radius: 8px; padding: 12px; }}
  h1 {{ color: {T['GOLD']} !important; }}
  h2 {{ color: {T['TEXT']} !important; border-bottom: 1px solid {T['BORDER']}; padding-bottom: 6px; }}
  h3 {{ color: {T['BLUE']} !important; }}
  hr {{ border-color: {T['BORDER']}; }}
  .stSelectbox label, .stMultiSelect label {{ color: {T['MUTED']} !important; font-size: 0.8rem; }}
  .info-box {{ background:{T['CARD']}; border-left:3px solid {T['GOLD']};
              padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; font-size:0.9rem; color:{T['TEXT']}; }}
  .warn-box {{ background:{T['CARD']}; border-left:3px solid {T['RED']};
              padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; font-size:0.9rem; color:{T['TEXT']}; }}
</style>
""", unsafe_allow_html=True)

apply_css()



# ── PALETA ───────────────────────────────────────────────────────────────────
GOLD   = T["GOLD"]
GREEN  = T["GREEN"]
BLUE   = T["BLUE"]
RED    = T["RED"]
CYAN   = T["CYAN"]
ORANGE = T["ORANGE"]
MUTED  = T["MUTED"]
BG     = T["BG"]
CARD   = T["CARD"]
BORDER = T["BORDER"]
TEXT   = T["TEXT"]

POLARITY_LABEL = {"bom": "✚ BOM", "mau": "✖ MAU"}
FASE_COLORS = [GREEN, CYAN, BLUE, ORANGE, GOLD, "#C084FC"]

# Cor por jogo (para que cada jogo tenha identidade visual consistente)
MATCH_COLORS = {
    "POR-HUN": GREEN,
    "POR-TUR": BLUE,
    "POR-GEO": GOLD,
    "POR-SVN": ORANGE,
    "POR-FRA": RED,
}


# ── HELPERS ──────────────────────────────────────────────────────────────────
def explode_field(df, col, sep=";"):
    d = df.copy()
    d[col] = d[col].fillna("").str.split(sep)
    d = d.explode(col)
    d[col] = d[col].str.strip()
    return d[d[col] != ""]


def theme(**overrides):
    """Devolve o tema base do plotly, com overrides opcionais aplicados correctamente.
    Evita o TypeError de keyword duplicado ao fundir xaxis/yaxis em vez de passar em duplicado."""
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font_color=TEXT,
        font_size=11,
        margin=dict(t=45, b=30, l=10, r=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )
    # Fundir xaxis/yaxis em vez de substituir (evita keyword duplicado)
    for k in ("xaxis", "yaxis"):
        if k in overrides:
            base[k] = {**base[k], **overrides.pop(k)}
    base.update(overrides)
    return base


@st.cache_data
def load_data(files=None):
    if files:
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        demo = "fpf_sessao_uefapro_demo.csv"
        if not os.path.exists(demo):
            st.error("Ficheiro de demonstração não encontrado. Executa `python gerar_dados_demo.py` primeiro.")
            st.stop()
        df = pd.read_csv(demo)

    df["minute"]       = pd.to_numeric(df["minute"],       errors="coerce")
    df["half"]         = pd.to_numeric(df["half"],         errors="coerce")
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    # Normalise polarity: the tool exports 'good'/'bad'; demo data uses 'bom'/'mau'
    pol_map = {"good": "bom", "bad": "mau", "bom": "bom", "mau": "mau"}
    df["polarity"] = df["polarity"].str.strip().str.lower().map(pol_map).fillna("mau")

    # Criar label de jogo legível: "POR-HUN vs Hungria"
    if "adversario" in df.columns:
        df["jogo_label"] = df["match_ref"] + " vs " + df["adversario"]
    else:
        df["jogo_label"] = df["match_ref"]

    return df




def multiselect_with_all(label, options, default_all=True, key=None, format_func=None):
    options = [o for o in options if o is not None]
    all_option = "(Todos)"
    opts = [all_option] + list(options)

    def _fmt(x):
        if x == all_option:
            return all_option
        return format_func(x) if format_func else x

    default = [all_option] if default_all else []
    selected = st.multiselect(label, opts, default=default, key=key, format_func=_fmt)
    if all_option in selected:
        return list(options)
    return selected
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h1 style='font-size:1.1rem; color:{GOLD};'>⚽ FPF · UEFA Pro<br>Análise de Sessão</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader(
        "CSV(s) dos grupos", type="csv", accept_multiple_files=True,
        help="Arrastar os CSVs exportados pelos grupos. Sem upload = dados de demonstração."
    )
    df_raw = load_data(uploaded if uploaded else None)

    st.markdown("---")
    st.markdown(f"<p style='color:{MUTED}; font-size:0.8rem; margin-bottom:4px;'>FILTROS</p>",
                unsafe_allow_html=True)

    all_groups  = sorted(df_raw["analyst"].dropna().unique())
    all_matches = sorted(df_raw["jogo_label"].dropna().unique())
    all_phases  = sorted(df_raw["phase"].dropna().unique())

    sel_groups  = multiselect_with_all("Grupos",     all_groups,  default_all=True, key="f_groups")
    sel_matches = multiselect_with_all("Jogos",      all_matches, default_all=True, key="f_matches")
    sel_phases  = multiselect_with_all("Fases",      all_phases,  default_all=True, key="f_phases")
    sel_pol     = multiselect_with_all("Polaridade", ["bom","mau"], default_all=True, key="f_pol",
                                       format_func=lambda x: POLARITY_LABEL[x])

    st.markdown("---")
    st.markdown(f"""<div style='font-size:0.75rem; color:{MUTED};'>
    <b style='color:{GOLD};'>Contexto</b><br>
    Cada grupo analisou a <b>1ª parte</b> de um jogo diferente.<br><br>
    <b style='color:{GOLD};'>Campos multi-valor</b><br>
    <code>categories</code> · <code>qualifiers</code><br>
    <code>players_involved</code><br>
    separados por <code>;</code><br><br>
    Rui Marcelino · FPF
    </div>""", unsafe_allow_html=True)


# ── FILTRAR ───────────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_groups:  df = df[df["analyst"].isin(sel_groups)]
if sel_matches: df = df[df["jogo_label"].isin(sel_matches)]
if sel_phases:  df = df[df["phase"].isin(sel_phases)]
if sel_pol:     df = df[df["polarity"].isin(sel_pol)]

df_cats    = explode_field(df, "categories")
df_quals   = explode_field(df, "qualifiers")

# players_involved_names is always ';'-separated and has the format "#13 Renato Veiga" or "Renato Veiga"
# Use it as the canonical source for player explosion.
# Fallback: if absent or empty, fall back to players_involved (may use ',' or ';' separator).
def explode_players(df):
    """Explode player names robustly regardless of source format."""
    d = df.copy()
    # Prefer players_involved_names (always ';' separated)
    if "players_involved_names" in d.columns:
        src_col = "players_involved_names"
        sep = ";"
    else:
        src_col = "players_involved"
        # Detect separator: real tool uses ', ', demo uses ';'
        sample = d[src_col].dropna().astype(str).str.cat(sep=" ")
        sep = ";" if ";" in sample else ","

    d["_player_raw"] = d[src_col].fillna("")
    d["_player_raw"] = d["_player_raw"].str.split(sep)
    d = d.explode("_player_raw")
    d["_player_raw"] = d["_player_raw"].str.strip()
    d = d[d["_player_raw"] != ""]

    # Parse "#13 Renato Veiga" → num=13, name="Renato Veiga"
    # Also handles plain "Renato Veiga" (no number prefix)
    import re
    def parse_player(raw):
        m = re.match(r"#?(\d+)\s+(.*)", str(raw).strip())
        if m:
            return int(m.group(1)), m.group(2).strip()
        return None, str(raw).strip()

    parsed = d["_player_raw"].apply(lambda x: pd.Series(parse_player(x),
                                                          index=["player_num","player_name"]))
    d["player_num"]  = parsed["player_num"]
    d["player_name"] = parsed["player_name"]
    d = d.drop(columns=["_player_raw"])
    return d[d["player_name"] != ""]

df_players = explode_players(df)

if len(df) == 0:
    st.warning("Nenhum dado com os filtros seleccionados.")
    st.stop()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='border-left:4px solid {GOLD}; padding:8px 16px; background:{CARD};
            border-radius:0 8px 8px 0; margin-bottom:16px;'>
  <h1 style='margin:0; font-size:1.6rem;'>O Treinador que Decide com Dados</h1>
  <p style='margin:4px 0 0; color:{MUTED}; font-size:0.9rem;'>
    5 grupos · 5 jogos · 1ª parte de cada · análise agregada de padrões e tendências
  </p>
</div>
""", unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────────────────────
n_total  = len(df)
n_bom    = (df["polarity"] == "bom").sum()
n_mau    = (df["polarity"] == "mau").sum()
pct_bom  = n_bom / n_total * 100 if n_total else 0
n_jogos  = df["match_ref"].nunique()
n_players= df_players["player_name"].nunique()
dur_med  = df["duration_sec"].mean()

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📋 Momentos", n_total)
c2.metric("✚ BOM",       f"{n_bom} ({pct_bom:.0f}%)")
c3.metric("✖ MAU",       f"{n_mau} ({100-pct_bom:.0f}%)")
c4.metric("⚽ Jogos",    n_jogos)
c5.metric("🧍 Jogadores",n_players)
c6.metric("⏱ Clip médio",f"{dur_med:.0f}s")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 1 — VISÃO GERAL: um jogo por grupo
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · Visão Geral — Um Jogo por Grupo")
st.markdown(
    f'<div class="info-box">Cada grupo analisou a <b>1ª parte de um jogo diferente</b>. '
    f'A comparação entre grupos é portanto uma comparação entre jogos e entre olhares. '
    f'Os padrões transversais emergem por agregação.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.3, 1])

with col1:
    # BOM/MAU por jogo (= por grupo)
    gp = (df.groupby(["jogo_label","match_ref","polarity"]).size()
            .unstack(fill_value=0).reset_index())
    if "bom" not in gp.columns: gp["bom"] = 0
    if "mau" not in gp.columns: gp["mau"] = 0
    gp["total"] = gp["bom"] + gp["mau"]
    gp["pct_bom"] = gp["bom"] / gp["total"].replace(0,1) * 100
    gp_sorted = gp.sort_values("pct_bom", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="✚ BOM", x=gp_sorted["jogo_label"], y=gp_sorted["bom"],
                         marker_color=GREEN, opacity=0.9,
                         text=gp_sorted["bom"], textposition="inside", textfont_color="#000"))
    fig.add_trace(go.Bar(name="✖ MAU", x=gp_sorted["jogo_label"], y=gp_sorted["mau"],
                         marker_color=RED, opacity=0.9,
                         text=gp_sorted["mau"], textposition="inside", textfont_color="#fff"))
    fig.update_layout(barmode="stack", title="Momentos por jogo (ordenado por % BOM)",
                      xaxis_tickangle=-20, **theme())
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # % BOM por jogo — lollipop horizontal
    gp_s = gp.sort_values("pct_bom")
    fig2 = go.Figure()
    for _, row in gp_s.iterrows():
        color = GREEN if row["pct_bom"] >= 50 else RED
        fig2.add_shape(type="line",
                       x0=50, x1=row["pct_bom"],
                       y0=row["jogo_label"], y1=row["jogo_label"],
                       line=dict(color=color, width=2))
        fig2.add_trace(go.Scatter(
            x=[row["pct_bom"]], y=[row["jogo_label"]],
            mode="markers+text",
            marker=dict(color=color, size=14),
            text=[f"{row['pct_bom']:.0f}%"],
            textposition="middle right",
            textfont=dict(color=GOLD, size=11),
            showlegend=False,
        ))
    fig2.add_vline(x=50, line_dash="dash", line_color=GOLD, annotation_text="50%",
                   annotation_font_color=GOLD)
    fig2.update_layout(title="% Momentos BOM por jogo",
                       xaxis_range=[20,100], xaxis_title="%",
                       **theme())
    st.plotly_chart(fig2, use_container_width=True)

# Tabela resumo
if "adversario" in df.columns and "competicao" in df.columns:
    resumo_tab = df.groupby(["analyst","match_ref","adversario","competicao","resultado"]).agg(
        Total=("polarity","count"),
        BOM=("polarity", lambda x: (x=="bom").sum()),
        MAU=("polarity", lambda x: (x=="mau").sum()),
    ).reset_index()
    resumo_tab["% BOM"] = (resumo_tab["BOM"] / resumo_tab["Total"] * 100).round(1).astype(str) + "%"
    st.dataframe(resumo_tab.rename(columns={"analyst":"Grupo","match_ref":"Jogo",
                                             "adversario":"Adversário","competicao":"Competição",
                                             "resultado":"Resultado"}),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 2 — DISTRIBUIÇÃO TEMPORAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 2 · Distribuição Temporal (1ª Parte)")

col_t1, col_t2 = st.columns([1.6, 1])

with col_t1:
    fig3 = go.Figure()
    for pol, color, label in [("bom", GREEN, "✚ BOM"), ("mau", RED, "✖ MAU")]:
        sub = df[df["polarity"] == pol]["minute"].dropna()
        fig3.add_trace(go.Histogram(x=sub, nbinsx=15, name=label,
                                    marker_color=color, opacity=0.75))
    fig3.update_layout(barmode="overlay", title="Distribuição dos momentos por minuto",
                       xaxis_title="Minuto", yaxis_title="Frequência",
                       **theme())
    st.plotly_chart(fig3, use_container_width=True)

with col_t2:
    # Momentos por jogo e minuto (scatter)
    tl = df.groupby(["match_ref","minute","polarity"]).size().reset_index(name="n")
    fig_tl = px.scatter(tl, x="minute", y="match_ref", size="n",
                        color="polarity", color_discrete_map={"bom":GREEN,"mau":RED},
                        size_max=22, labels={"match_ref":"Jogo","minute":"Minuto","polarity":""},
                        title="Linha temporal por jogo")
    fig_tl.update_layout(**theme(), height=280)
    st.plotly_chart(fig_tl, use_container_width=True)

# Intensidade minuto a minuto (todos os jogos sobrepostos)
st.markdown("#### Intensidade por minuto — todos os jogos sobrepostos")
minute_pol = df.groupby(["minute","polarity"]).size().reset_index(name="n")
fig_int = go.Figure()
for pol, color, label in [("bom",GREEN,"✚ BOM"),("mau",RED,"✖ MAU")]:
    sub = minute_pol[minute_pol["polarity"]==pol]
    fig_int.add_trace(go.Scatter(x=sub["minute"], y=sub["n"], mode="lines+markers",
                                 name=label, line=dict(color=color, width=2),
                                 marker=dict(size=5), fill="tozeroy",
                                 fillcolor=color.replace("#","rgba(").replace("",",0.1)") if False else
                                 f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"))
fig_int.update_layout(title="Acumulado de momentos por minuto (todos os jogos)",
                      xaxis_title="Minuto", yaxis_title="Nº de momentos",
                      **theme())
st.plotly_chart(fig_int, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 3 — ANÁLISE POR FASE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 3 · Análise por Fase de Jogo")

col_f1, col_f2 = st.columns(2)

with col_f1:
    fp = (df.groupby(["phase","polarity"]).size().unstack(fill_value=0).reset_index())
    if "bom" not in fp.columns: fp["bom"] = 0
    if "mau" not in fp.columns: fp["mau"] = 0
    fp["total"] = fp["bom"] + fp["mau"]
    fp = fp.sort_values("total", ascending=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(name="✚ BOM", y=fp["phase"], x=fp["bom"],
                          orientation="h", marker_color=GREEN, opacity=0.9))
    fig5.add_trace(go.Bar(name="✖ MAU", y=fp["phase"], x=fp["mau"],
                          orientation="h", marker_color=RED, opacity=0.9))
    fig5.update_layout(barmode="stack", title="Volume por fase",
                       xaxis_title="Momentos", **theme(), height=350)
    st.plotly_chart(fig5, use_container_width=True)

with col_f2:
    # Fases por jogo — heatmap (% do total de cada jogo)
    fj = (df.groupby(["match_ref","phase"]).size().unstack(fill_value=0))
    fj_pct = fj.div(fj.sum(axis=1), axis=0) * 100
    fig_fj = px.imshow(fj_pct.round(1),
                       color_continuous_scale="Blues",
                       title="Distribuição de fases por jogo (%)",
                       labels=dict(x="Fase", y="Jogo", color="%"),
                       aspect="auto", text_auto=".0f")
    fig_fj.update_layout(**theme(), height=350)
    fig_fj.update_xaxes(tickangle=30, tickfont_size=9)
    st.plotly_chart(fig_fj, use_container_width=True)

# Radar — perfil BOM vs MAU por fase
fases_list = sorted(df["phase"].unique())
bom_r = [((df["phase"]==f) & (df["polarity"]=="bom")).sum() for f in fases_list]
mau_r = [((df["phase"]==f) & (df["polarity"]=="mau")).sum() for f in fases_list]

fig6 = go.Figure()
fig6.add_trace(go.Scatterpolar(
    r=bom_r+[bom_r[0]], theta=fases_list+[fases_list[0]],
    fill="toself", name="✚ BOM",
    line_color=GREEN, fillcolor="rgba(59,219,106,0.12)"))
fig6.add_trace(go.Scatterpolar(
    r=mau_r+[mau_r[0]], theta=fases_list+[fases_list[0]],
    fill="toself", name="✖ MAU",
    line_color=RED, fillcolor="rgba(232,72,85,0.12)"))
fig6.update_layout(
    polar=dict(bgcolor=CARD,
               radialaxis=dict(visible=True, color=MUTED, gridcolor=BORDER),
               angularaxis=dict(color="#C8D6EE")),
    title="Perfil radar BOM vs MAU (todos os jogos agregados)",
    **theme(), height=380)
st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 4 — CATEGORIAS (multi-valor)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 4 · Categorias")
st.markdown(
    f'<div class="info-box">⚠️ <b>Campo multi-valor</b>: cada momento pode ter várias categorias '
    f'separadas por <code>;</code>. A soma das ocorrências pode exceder o total de momentos.</div>',
    unsafe_allow_html=True
)

top_n = st.slider("Top N categorias", 5, 15, 10, key="cats_n")

col_c1, col_c2 = st.columns(2)

for col, pol, color, label in [
    (col_c1, "bom", GREEN, "✚ Momentos BOM"),
    (col_c2, "mau", RED,   "✖ Momentos MAU"),
]:
    with col:
        sub = df_cats[df_cats["polarity"] == pol]["categories"].value_counts().head(top_n)
        fig_c = go.Figure(go.Bar(
            y=sub.index, x=sub.values, orientation="h",
            marker_color=color, opacity=0.85,
            text=sub.values, textposition="outside", textfont_color=TEXT,
        ))
        # Correcto: yaxis passado via theme() com override fundido
        fig_c.update_layout(
            title=label,
            xaxis_title="Ocorrências",
            **theme(yaxis={"autorange": "reversed"}),
            height=380
        )
        st.plotly_chart(fig_c, use_container_width=True)

# Categorias por jogo — comparação transversal
st.markdown("#### Categorias por jogo — que temas emergem em cada adversário?")
top_cats_global = df_cats["categories"].value_counts().head(8).index.tolist()
cats_jogo = (df_cats[df_cats["categories"].isin(top_cats_global)]
             .groupby(["match_ref","categories"]).size().reset_index(name="n"))

match_labels = df[["match_ref","jogo_label"]].drop_duplicates()
cats_jogo = cats_jogo.merge(match_labels, on="match_ref", how="left")

fig_cj = px.bar(cats_jogo, x="categories", y="n", color="jogo_label",
                barmode="group",
                color_discrete_sequence=list(MATCH_COLORS.values()),
                labels={"categories":"Categoria","n":"Ocorrências","jogo_label":"Jogo"},
                title="Top categorias por jogo")
fig_cj.update_layout(**theme(xaxis={"tickangle": -25}))
st.plotly_chart(fig_cj, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 5 — JOGADORES (multi-valor)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 5 · Jogadores Mais Envolvidos")
st.markdown(
    f'<div class="info-box">⚠️ <b>Campo multi-valor</b>: cada momento pode envolver vários jogadores '
    f'(formato <code>NÚM-Nome</code> separados por <code>;</code>).</div>',
    unsafe_allow_html=True
)

top_np = st.slider("Top N jogadores", 5, 18, 12, key="players_n")

pp = (df_players.groupby(["player_name","polarity"]).size()
      .unstack(fill_value=0).reset_index())
if "bom" not in pp.columns: pp["bom"] = 0
if "mau" not in pp.columns: pp["mau"] = 0
pp["total"] = pp["bom"] + pp["mau"]
pp["ratio_mau"] = pp["mau"] / pp["total"].replace(0,1)
pp = pp.sort_values("total", ascending=False).head(top_np)

col_p1, col_p2 = st.columns(2)

with col_p1:
    pp_s = pp.sort_values("total", ascending=True)
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(name="✚ BOM", y=pp_s["player_name"], x=pp_s["bom"],
                           orientation="h", marker_color=GREEN, opacity=0.9))
    fig_p.add_trace(go.Bar(name="✖ MAU", y=pp_s["player_name"], x=pp_s["mau"],
                           orientation="h", marker_color=RED, opacity=0.9))
    fig_p.update_layout(barmode="stack", title="Menções por jogador",
                        **theme(), height=420)
    st.plotly_chart(fig_p, use_container_width=True)

with col_p2:
    fig_ps = px.scatter(
        pp, x="total", y="ratio_mau", text="player_name",
        color="ratio_mau",
        color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],
        size="total", size_max=28,
        labels={"total":"Total menções","ratio_mau":"Proporção MAU","player_name":""},
        title="Visibilidade vs Negatividade",
    )
    fig_ps.add_hline(y=0.5, line_dash="dash", line_color=GOLD,
                     annotation_text="50%", annotation_font_color=GOLD)
    fig_ps.update_traces(textposition="top center", textfont_size=9)
    fig_ps.update_layout(**theme(), height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_ps, use_container_width=True)

# Jogador nos diferentes jogos
st.markdown("#### Presença por jogo — jogadores transversais aos 5 jogos")
player_jogo = (df_players.groupby(["player_name","match_ref","polarity"]).size()
               .reset_index(name="n"))
top_players = df_players["player_name"].value_counts().head(10).index.tolist()
pj_top = player_jogo[player_jogo["player_name"].isin(top_players)]
fig_pj = px.bar(pj_top, x="player_name", y="n", color="polarity",
                facet_col="match_ref", facet_col_wrap=5,
                color_discrete_map={"bom":GREEN,"mau":RED},
                labels={"player_name":"","n":"Menções","polarity":""},
                title="Top 10 jogadores por jogo (BOM vs MAU)")
fig_pj.update_layout(**theme(), height=300)
fig_pj.update_xaxes(tickangle=45, tickfont_size=8)
st.plotly_chart(fig_pj, use_container_width=True)

# Detalhe individual
st.markdown("#### Detalhe de um jogador")
all_pnames = sorted(df_players["player_name"].dropna().unique())
if all_pnames:
    sel_p = st.selectbox("Seleciona jogador", all_pnames)
    sub_p = df_players[df_players["player_name"] == sel_p]
    np_total = len(sub_p)
    np_bom   = (sub_p["polarity"]=="bom").sum()

    p1,p2,p3 = st.columns(3)
    p1.metric("Total menções", np_total)
    p2.metric("✚ BOM", f"{np_bom} ({np_bom/np_total*100:.0f}%)" if np_total else "—")
    p3.metric("✖ MAU", f"{np_total-np_bom} ({(np_total-np_bom)/np_total*100:.0f}%)" if np_total else "—")

    col_pd1, col_pd2 = st.columns(2)
    with col_pd1:
        fp2 = sub_p.groupby(["phase","polarity"]).size().reset_index(name="n")
        fig_pd = px.bar(fp2, x="phase", y="n", color="polarity",
                        color_discrete_map={"bom":GREEN,"mau":RED}, barmode="group",
                        title=f"{sel_p} — por fase",
                        labels={"phase":"","n":"Menções","polarity":""})
        fig_pd.update_layout(**theme(xaxis={"tickangle":-25}))
        st.plotly_chart(fig_pd, use_container_width=True)
    with col_pd2:
        mj2 = sub_p.groupby(["match_ref","polarity"]).size().reset_index(name="n")
        fig_mj = px.bar(mj2, x="match_ref", y="n", color="polarity",
                        color_discrete_map={"bom":GREEN,"mau":RED}, barmode="group",
                        title=f"{sel_p} — por jogo",
                        labels={"match_ref":"","n":"Menções","polarity":""})
        fig_mj.update_layout(**theme())
        st.plotly_chart(fig_mj, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 6 — ZONA DO CAMPO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 6 · Zona do Campo")

col_z1, col_z2 = st.columns(2)
for col, pol, cscale, title in [
    (col_z1, "bom", "Greens", "✚ Momentos BOM"),
    (col_z2, "mau", "Reds",   "✖ Momentos MAU"),
]:
    with col:
        zf = (df[df["polarity"]==pol]
              .groupby(["zone","phase"]).size().unstack(fill_value=0))
        if zf.empty:
            st.info(f"Sem dados ({title})")
            continue
        fig_z = px.imshow(zf, color_continuous_scale=cscale, title=title,
                          labels=dict(x="Fase",y="Zona",color="N"),
                          aspect="auto", text_auto=True)
        fig_z.update_layout(**theme(), height=380)
        fig_z.update_xaxes(tickangle=30, tickfont_size=9)
        st.plotly_chart(fig_z, use_container_width=True)

# Zona por jogo
st.markdown("#### Zonas de intervenção por jogo")
zona_jogo = df.groupby(["match_ref","zone"]).size().unstack(fill_value=0)
zona_jogo_pct = zona_jogo.div(zona_jogo.sum(axis=1), axis=0) * 100
fig_zj = px.imshow(zona_jogo_pct.round(1),
                   color_continuous_scale="Viridis",
                   title="Distribuição de zonas por jogo (%)",
                   labels=dict(x="Zona",y="Jogo",color="%"),
                   aspect="auto", text_auto=".0f")
fig_zj.update_layout(**theme())
fig_zj.update_xaxes(tickangle=35, tickfont_size=9)
st.plotly_chart(fig_zj, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 7 — QUALIFICADORES
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 7 · Qualificadores")

col_q1, col_q2 = st.columns(2)
for col, pol, color, label in [
    (col_q1, "bom", GREEN, "✚ Momentos BOM"),
    (col_q2, "mau", RED,   "✖ Momentos MAU"),
]:
    with col:
        q_counts = (df_quals[df_quals["polarity"]==pol]["qualifiers"]
                    .value_counts().head(10))
        fig_q = go.Figure(go.Bar(
            y=q_counts.index, x=q_counts.values, orientation="h",
            marker_color=color, opacity=0.85,
            text=q_counts.values, textposition="outside",
        ))
        fig_q.update_layout(
            title=label,
            **theme(yaxis={"autorange": "reversed"}),
            height=360
        )
        st.plotly_chart(fig_q, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 8 — PADRÕES TRANSVERSAIS (o coração da análise)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 8 · Padrões Transversais aos 5 Jogos")
st.markdown(
    f'<div class="info-box">Esta é a secção central: o que é <b>consistente</b> em todos os jogos? '
    f'O que varia com o adversário? Que tendências atravessam os 5 primeiros tempos?</div>',
    unsafe_allow_html=True
)

col_pt1, col_pt2 = st.columns(2)

with col_pt1:
    # % BOM por fase e por jogo — heatmap
    fase_jogo_bom = (df.groupby(["match_ref","phase","polarity"]).size()
                     .unstack(fill_value=0).reset_index())
    if "bom" not in fase_jogo_bom.columns: fase_jogo_bom["bom"] = 0
    if "mau" not in fase_jogo_bom.columns: fase_jogo_bom["mau"] = 0
    fase_jogo_bom["total"] = fase_jogo_bom["bom"] + fase_jogo_bom["mau"]
    fase_jogo_bom["pct_bom"] = (fase_jogo_bom["bom"] /
                                  fase_jogo_bom["total"].replace(0,1) * 100)
    piv = fase_jogo_bom.pivot(index="match_ref", columns="phase", values="pct_bom").fillna(0)
    fig_hm = px.imshow(piv.round(1),
                       color_continuous_scale=[[0,"#2A080A"],[0.5,"#3A3010"],[1,"#0F2E1A"]],
                       zmin=0, zmax=100,
                       title="% BOM por Fase × Jogo",
                       labels=dict(x="Fase",y="Jogo",color="% BOM"),
                       aspect="auto", text_auto=".0f")
    fig_hm.update_layout(**theme(), height=320)
    fig_hm.update_xaxes(tickangle=30, tickfont_size=9)
    st.plotly_chart(fig_hm, use_container_width=True)

with col_pt2:
    # Tendência por minuto — BOM - MAU (saldo)
    saldo = df.groupby("minute").apply(
        lambda x: (x["polarity"]=="bom").sum() - (x["polarity"]=="mau").sum()
    ).reset_index(name="saldo")
    saldo = saldo.sort_values("minute")
    saldo["color"] = saldo["saldo"].apply(lambda v: GREEN if v >= 0 else RED)

    fig_sal = go.Figure(go.Bar(
        x=saldo["minute"], y=saldo["saldo"],
        marker_color=saldo["color"], opacity=0.85,
    ))
    fig_sal.add_hline(y=0, line_color=GOLD, line_dash="dash")
    fig_sal.update_layout(title="Saldo BOM−MAU por minuto (acumulado todos os jogos)",
                          xaxis_title="Minuto", yaxis_title="BOM − MAU",
                          **theme(), height=320)
    st.plotly_chart(fig_sal, use_container_width=True)

# Top categorias comuns a todos os jogos
st.markdown("#### Categorias presentes em todos (ou quase todos) os jogos")
cats_por_jogo = (df_cats.groupby(["categories","match_ref"]).size()
                 .reset_index(name="n"))
cats_njogos = cats_por_jogo.groupby("categories")["match_ref"].nunique().reset_index()
cats_njogos.columns = ["categoria","n_jogos"]
cats_com_freq = (df_cats.groupby("categories").size().reset_index(name="total"))
cats_njogos = cats_njogos.merge(cats_com_freq, left_on="categoria", right_on="categories")
cats_njogos = cats_njogos.sort_values(["n_jogos","total"], ascending=False).head(12)

fig_cta = px.scatter(cats_njogos, x="total", y="n_jogos",
                     text="categoria",
                     size="total", size_max=24,
                     color="n_jogos",
                     color_continuous_scale=[[0,MUTED],[0.5,BLUE],[1,GREEN]],
                     labels={"total":"Total ocorrências","n_jogos":"Nº de jogos em que aparece"},
                     title="Categorias mais transversais (presentes em mais jogos)")
fig_cta.update_traces(textposition="top center", textfont_size=9)
fig_cta.update_layout(**theme(), coloraxis_showscale=False)
st.plotly_chart(fig_cta, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 9 — EXPLORADOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 9 · Explorador de Momentos")

c_e1, c_e2, c_e3, c_e4 = st.columns(4)
pol_ex   = c_e1.selectbox("Polaridade", ["Todos","bom","mau"],
                            format_func=lambda x: POLARITY_LABEL.get(x,x))
fase_ex  = c_e2.selectbox("Fase", ["Todas"]+sorted(df["phase"].unique()))
jogo_ex  = c_e3.selectbox("Jogo", ["Todos"]+sorted(df["match_ref"].unique()))
prio_ex  = c_e4.selectbox("Prioridade", ["Todas"]+sorted(df["priority"].dropna().unique()))

df_ex = df.copy()
if pol_ex  != "Todos": df_ex = df_ex[df_ex["polarity"]==pol_ex]
if fase_ex != "Todas": df_ex = df_ex[df_ex["phase"]==fase_ex]
if jogo_ex != "Todos": df_ex = df_ex[df_ex["match_ref"]==jogo_ex]
if prio_ex != "Todas": df_ex = df_ex[df_ex["priority"]==prio_ex]

cols_show = ["analyst","match_ref","minute","polarity","phase","categories",
             "qualifiers","zone","players_involved_names","priority","description"]
cols_ok = [c for c in cols_show if c in df_ex.columns]

st.dataframe(df_ex[cols_ok].reset_index(drop=True),
             use_container_width=True, height=340,
             column_config={
                 "polarity": st.column_config.TextColumn("Polar."),
                 "minute":   st.column_config.NumberColumn("Min.", format="%d'"),
                 "analyst":  st.column_config.TextColumn("Grupo"),
                 "match_ref":st.column_config.TextColumn("Jogo"),
                 "priority": st.column_config.TextColumn("Prior."),
                 "players_involved_names": st.column_config.TextColumn("Jogadores"),
                 "description": st.column_config.TextColumn("Descrição", width="large"),
             })


# ── RODAPÉ ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:{MUTED}; font-size:0.82rem; padding:8px;'>
  FPF · Curso UEFA Pro · Análise de Desempenho ·
  <span style='color:{GOLD};'>Rui Marcelino</span> · rui.marcelino@fpf.pt<br>
  <span style='color:#252E48;'>Schema: FPF Video Tagger v2026.3 · 
  5 grupos · 5 jogos · 1ª parte de cada</span>
</div>
""", unsafe_allow_html=True)
