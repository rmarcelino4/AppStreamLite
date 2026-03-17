"""
FPF Video Tagger · Dashboard de Análise — Sessão UEFA Pro (Altair)
=================================================================
streamlit run fpf_dashboard-2_auth_altair.py

Requisitos: pip install streamlit pandas numpy altair
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
    index=0,
    key="style_choice",
)

T = THEMES["dark" if style_choice == "Escuro" else "light"]

# ── CSS ───────────────────────────────────────────────────────────────────

def apply_css():
    st.markdown(
        f"""
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
""",
        unsafe_allow_html=True,
    )

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

MATCH_COLORS = {
    "POR-HUN": GREEN,
    "POR-TUR": BLUE,
    "POR-GEO": GOLD,
    "POR-SVN": ORANGE,
    "POR-FRA": RED,
}

alt.themes.register(
    "uefapro",
    lambda: {
        "config": {
            "background": BG,
            "view": {"stroke": BORDER},
            "title": {"color": TEXT, "fontSize": 14},
            "axis": {
                "labelColor": TEXT,
                "titleColor": TEXT,
                "gridColor": BORDER,
                "tickColor": BORDER,
            },
            "legend": {"labelColor": TEXT, "titleColor": TEXT},
        }
    },
)
alt.themes.enable("uefapro")

# ── HELPERS ─────────────────────────────────────────────────────────────────-

def explode_field(df, col, sep=";"):
    d = df.copy()
    d[col] = d[col].fillna("").str.split(sep)
    d = d.explode(col)
    d[col] = d[col].str.strip()
    return d[d[col] != ""]


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

    pol_map = {"good": "bom", "bad": "mau", "bom": "bom", "mau": "mau"}
    df["polarity"] = df["polarity"].str.strip().str.lower().map(pol_map).fillna("mau")

    if "adversario" in df.columns:
        df["jogo_label"] = df["match_ref"] + " vs " + df["adversario"]
    else:
        df["jogo_label"] = df["match_ref"]

    return df


# players_involved_names is always ';'-separated and has the format "#13 Renato Veiga" or "Renato Veiga"
# Use it as the canonical source for player explosion.
# Fallback: if absent or empty, fall back to players_involved (may use ',' or ';' separator).

def explode_players(df):
    d = df.copy()
    if "players_involved_names" in d.columns:
        src_col = "players_involved_names"
        sep = ";"
    else:
        src_col = "players_involved"
        sample = d[src_col].dropna().astype(str).str.cat(sep=" ")
        sep = ";" if ";" in sample else ","

    d["_player_raw"] = d[src_col].fillna("")
    d["_player_raw"] = d["_player_raw"].str.split(sep)
    d = d.explode("_player_raw")
    d["_player_raw"] = d["_player_raw"].str.strip()
    d = d[d["_player_raw"] != ""]

    import re

    def parse_player(raw):
        m = re.match(r"#?(\d+)\s+(.*)", str(raw).strip())
        if m:
            return int(m.group(1)), m.group(2).strip()
        return None, str(raw).strip()

    parsed = d["_player_raw"].apply(lambda x: pd.Series(parse_player(x), index=["player_num", "player_name"]))
    d["player_num"] = parsed["player_num"]
    d["player_name"] = parsed["player_name"]
    d = d.drop(columns=["_player_raw"])
    return d[d["player_name"] != ""]


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

    sel_groups  = multiselect_with_all("Grupos", all_groups, default_all=True, key="f_groups")
    sel_matches = multiselect_with_all("Jogos", all_matches, default_all=True, key="f_matches")
    sel_phases  = multiselect_with_all("Fases", all_phases, default_all=True, key="f_phases")
    sel_pol     = multiselect_with_all("Polaridade", ["bom", "mau"], default_all=True, key="f_pol",
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

if len(df) == 0:
    st.warning("Nenhum dado com os filtros seleccionados.")
    st.stop()


df_cats    = explode_field(df, "categories")
df_quals   = explode_field(df, "qualifiers")
df_players = explode_players(df)


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
# SECÇÃO 1 — VISÃO GERAL
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
    gp = (df.groupby(["jogo_label", "match_ref", "polarity"]).size()
            .unstack(fill_value=0).reset_index())
    if "bom" not in gp.columns: gp["bom"] = 0
    if "mau" not in gp.columns: gp["mau"] = 0
    gp["total"] = gp["bom"] + gp["mau"]
    gp["pct_bom"] = gp["bom"] / gp["total"].replace(0, 1) * 100
    gp_sorted = gp.sort_values("pct_bom", ascending=False)

    data_m = gp_sorted.melt(id_vars=["jogo_label"], value_vars=["bom", "mau"],
                            var_name="polarity", value_name="n")
    color_scale = alt.Scale(domain=["bom", "mau"], range=[GREEN, RED])

    chart = alt.Chart(data_m).mark_bar().encode(
        x=alt.X("n:Q", stack=True, title="Momentos"),
        y=alt.Y("jogo_label:N", sort=gp_sorted["jogo_label"].tolist(), title=""),
        color=alt.Color("polarity:N", scale=color_scale, legend=alt.Legend(title="")),
        tooltip=["jogo_label", "polarity", "n"],
    ).properties(title="Momentos por jogo (ordenado por % BOM)")

    st.altair_chart(chart, use_container_width=True)

with col2:
    gp_s = gp.sort_values("pct_bom")
    chart = alt.Chart(gp_s).mark_bar(color=GREEN).encode(
        x=alt.X("pct_bom:Q", title="% BOM", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("jogo_label:N", sort=gp_s["jogo_label"].tolist(), title=""),
        tooltip=["jogo_label", alt.Tooltip("pct_bom:Q", format=".0f")],
    ).properties(title="% Momentos BOM por jogo")
    st.altair_chart(chart, use_container_width=True)

if "adversario" in df.columns and "competicao" in df.columns:
    resumo_tab = df.groupby(["analyst", "match_ref", "adversario", "competicao", "resultado"]).agg(
        Total=("polarity", "count"),
        BOM=("polarity", lambda x: (x == "bom").sum()),
        MAU=("polarity", lambda x: (x == "mau").sum()),
    ).reset_index()
    resumo_tab["% BOM"] = (resumo_tab["BOM"] / resumo_tab["Total"] * 100).round(1).astype(str) + "%"
    st.dataframe(resumo_tab.rename(columns={
        "analyst": "Grupo",
        "match_ref": "Jogo",
        "adversario": "Adversário",
        "competicao": "Competição",
        "resultado": "Resultado",
    }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 2 — DISTRIBUIÇÃO TEMPORAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 2 · Distribuição Temporal (1ª Parte)")

col_t1, col_t2 = st.columns([1.6, 1])

with col_t1:
    chart = alt.Chart(df.dropna(subset=["minute"]))\
        .mark_bar(opacity=0.7)\
        .encode(
            x=alt.X("minute:Q", bin=alt.Bin(maxbins=15), title="Minuto"),
            y=alt.Y("count():Q", title="Frequência"),
            color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED]))
        ).properties(title="Distribuição dos momentos por minuto")
    st.altair_chart(chart, use_container_width=True)

with col_t2:
    tl = df.groupby(["match_ref", "minute", "polarity"]).size().reset_index(name="n")
    chart = alt.Chart(tl).mark_circle().encode(
        x=alt.X("minute:Q", title="Minuto"),
        y=alt.Y("match_ref:N", title="Jogo"),
        size=alt.Size("n:Q", scale=alt.Scale(range=[20, 300])),
        color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED]))
    ).properties(title="Linha temporal por jogo")
    st.altair_chart(chart, use_container_width=True)

st.markdown("#### Intensidade por minuto — todos os jogos sobrepostos")
minute_pol = df.groupby(["minute", "polarity"]).size().reset_index(name="n")
chart = alt.Chart(minute_pol).mark_line(point=True).encode(
    x=alt.X("minute:Q", title="Minuto"),
    y=alt.Y("n:Q", title="Nº de momentos"),
    color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED])),
).properties(title="Acumulado de momentos por minuto (todos os jogos)")
st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 3 — ANÁLISE POR FASE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 3 · Análise por Fase de Jogo")

col_f1, col_f2 = st.columns(2)

with col_f1:
    fp = (df.groupby(["phase", "polarity"]).size().unstack(fill_value=0).reset_index())
    if "bom" not in fp.columns: fp["bom"] = 0
    if "mau" not in fp.columns: fp["mau"] = 0
    fp["total"] = fp["bom"] + fp["mau"]
    fp = fp.sort_values("total", ascending=True)

    data_m = fp.melt(id_vars=["phase"], value_vars=["bom", "mau"],
                     var_name="polarity", value_name="n")
    chart = alt.Chart(data_m).mark_bar().encode(
        x=alt.X("n:Q", stack=True, title="Momentos"),
        y=alt.Y("phase:N", sort=fp["phase"].tolist(), title=""),
        color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED]))
    ).properties(title="Volume por fase")
    st.altair_chart(chart, use_container_width=True)

with col_f2:
    fj = (df.groupby(["match_ref", "phase"]).size().unstack(fill_value=0))
    fj_pct = fj.div(fj.sum(axis=1), axis=0) * 100
    fj_pct = fj_pct.reset_index().melt(id_vars=["match_ref"], var_name="phase", value_name="pct")
    chart = alt.Chart(fj_pct).mark_rect().encode(
        x=alt.X("phase:N", title="Fase"),
        y=alt.Y("match_ref:N", title="Jogo"),
        color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues"), title="%"),
        tooltip=["match_ref", "phase", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(title="Distribuição de fases por jogo (%)")
    st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 4 — CATEGORIAS
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
        data = sub.reset_index()
        data.columns = ["category", "n"]
        chart = alt.Chart(data).mark_bar(color=color).encode(
            x=alt.X("n:Q", title="Ocorrências"),
            y=alt.Y("category:N", sort="-x", title=""),
            tooltip=["category", "n"],
        ).properties(title=label)
        st.altair_chart(chart, use_container_width=True)

st.markdown("#### Categorias por jogo — que temas emergem em cada adversário?")
top_cats_global = df_cats["categories"].value_counts().head(8).index.tolist()
cats_jogo = (df_cats[df_cats["categories"].isin(top_cats_global)]
             .groupby(["match_ref", "categories"]).size().reset_index(name="n"))
match_labels = df[["match_ref", "jogo_label"]].drop_duplicates()
cats_jogo = cats_jogo.merge(match_labels, on="match_ref", how="left")

chart = alt.Chart(cats_jogo).mark_bar().encode(
    x=alt.X("categories:N", title="Categoria"),
    y=alt.Y("n:Q", title="Ocorrências"),
    color=alt.Color("jogo_label:N", title="Jogo"),
    column=alt.Column("jogo_label:N", header=alt.Header(title="")),
).properties(title="Top categorias por jogo")

st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 5 — JOGADORES
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 5 · Jogadores Mais Envolvidos")
st.markdown(
    f'<div class="info-box">⚠️ <b>Campo multi-valor</b>: cada momento pode envolver vários jogadores '
    f'(formato <code>NÚM-Nome</code> separados por <code>;</code>).</div>',
    unsafe_allow_html=True
)

top_np = st.slider("Top N jogadores", 5, 18, 12, key="players_n")

pp = (df_players.groupby(["player_name", "polarity"]).size()
      .unstack(fill_value=0).reset_index())
if "bom" not in pp.columns: pp["bom"] = 0
if "mau" not in pp.columns: pp["mau"] = 0
pp["total"] = pp["bom"] + pp["mau"]
pp["ratio_mau"] = pp["mau"] / pp["total"].replace(0, 1)
pp = pp.sort_values("total", ascending=False).head(top_np)

col_p1, col_p2 = st.columns(2)

with col_p1:
    data_m = pp.melt(id_vars=["player_name"], value_vars=["bom", "mau"],
                     var_name="polarity", value_name="n")
    chart = alt.Chart(data_m).mark_bar().encode(
        x=alt.X("n:Q", stack=True, title="Menções"),
        y=alt.Y("player_name:N", sort=pp["player_name"].tolist(), title=""),
        color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED]))
    ).properties(title="Menções por jogador")
    st.altair_chart(chart, use_container_width=True)

with col_p2:
    chart = alt.Chart(pp).mark_circle().encode(
        x=alt.X("total:Q", title="Total menções"),
        y=alt.Y("ratio_mau:Q", title="Proporção MAU"),
        size=alt.Size("total:Q", scale=alt.Scale(range=[50, 800])),
        color=alt.Color("ratio_mau:Q", scale=alt.Scale(domain=[0, 1], range=[GREEN, RED])),
        tooltip=["player_name", "total", alt.Tooltip("ratio_mau:Q", format=".2f")],
    ).properties(title="Visibilidade vs Negatividade")
    st.altair_chart(chart, use_container_width=True)

st.markdown("#### Presença por jogo — jogadores transversais aos 5 jogos")
player_jogo = (df_players.groupby(["player_name", "match_ref", "polarity"]).size()
               .reset_index(name="n"))
top_players = df_players["player_name"].value_counts().head(10).index.tolist()
pj_top = player_jogo[player_jogo["player_name"].isin(top_players)]

chart = alt.Chart(pj_top).mark_bar().encode(
    x=alt.X("player_name:N", title=""),
    y=alt.Y("n:Q", title="Menções"),
    color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED])),
    column=alt.Column("match_ref:N", header=alt.Header(title="Jogo")),
).properties(title="Top 10 jogadores por jogo (BOM vs MAU)")

st.altair_chart(chart, use_container_width=True)

st.markdown("#### Detalhe de um jogador")
all_pnames = sorted(df_players["player_name"].dropna().unique())
if all_pnames:
    sel_p = st.selectbox("Seleciona jogador", all_pnames)
    sub_p = df_players[df_players["player_name"] == sel_p]
    np_total = len(sub_p)
    np_bom   = (sub_p["polarity"] == "bom").sum()

    p1, p2, p3 = st.columns(3)
    p1.metric("Total menções", np_total)
    p2.metric("✚ BOM", f"{np_bom} ({np_bom/np_total*100:.0f}%)" if np_total else "—")
    p3.metric("✖ MAU", f"{np_total-np_bom} ({(np_total-np_bom)/np_total*100:.0f}%)" if np_total else "—")

    col_pd1, col_pd2 = st.columns(2)
    with col_pd1:
        fp2 = sub_p.groupby(["phase", "polarity"]).size().reset_index(name="n")
        chart = alt.Chart(fp2).mark_bar().encode(
            x=alt.X("phase:N", title=""),
            y=alt.Y("n:Q", title="Menções"),
            color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED])),
        ).properties(title=f"{sel_p} — por fase")
        st.altair_chart(chart, use_container_width=True)

    with col_pd2:
        mj2 = sub_p.groupby(["match_ref", "polarity"]).size().reset_index(name="n")
        chart = alt.Chart(mj2).mark_bar().encode(
            x=alt.X("match_ref:N", title=""),
            y=alt.Y("n:Q", title="Menções"),
            color=alt.Color("polarity:N", scale=alt.Scale(domain=["bom", "mau"], range=[GREEN, RED])),
        ).properties(title=f"{sel_p} — por jogo")
        st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 6 — ZONA DO CAMPO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 6 · Zona do Campo")

col_z1, col_z2 = st.columns(2)
for col, pol, scheme, title in [
    (col_z1, "bom", "greens", "✚ Momentos BOM"),
    (col_z2, "mau", "reds",   "✖ Momentos MAU"),
]:
    with col:
        zf = (df[df["polarity"] == pol]
              .groupby(["zone", "phase"]).size().reset_index(name="n"))
        if zf.empty:
            st.info(f"Sem dados ({title})")
            continue
        chart = alt.Chart(zf).mark_rect().encode(
            x=alt.X("phase:N", title="Fase"),
            y=alt.Y("zone:N", title="Zona"),
            color=alt.Color("n:Q", scale=alt.Scale(scheme=scheme), title="N"),
            tooltip=["zone", "phase", "n"],
        ).properties(title=title)
        st.altair_chart(chart, use_container_width=True)

st.markdown("#### Zonas de intervenção por jogo")
zona_jogo = df.groupby(["match_ref", "zone"]).size().unstack(fill_value=0)
zona_jogo_pct = zona_jogo.div(zona_jogo.sum(axis=1), axis=0) * 100
zj = zona_jogo_pct.reset_index().melt(id_vars=["match_ref"], var_name="zone", value_name="pct")

chart = alt.Chart(zj).mark_rect().encode(
    x=alt.X("zone:N", title="Zona"),
    y=alt.Y("match_ref:N", title="Jogo"),
    color=alt.Color("pct:Q", scale=alt.Scale(scheme="viridis"), title="%"),
    tooltip=["match_ref", "zone", alt.Tooltip("pct:Q", format=".1f")],
).properties(title="Distribuição de zonas por jogo (%)")

st.altair_chart(chart, use_container_width=True)


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
        q_counts = (df_quals[df_quals["polarity"] == pol]["qualifiers"]
                    .value_counts().head(10))
        data = q_counts.reset_index()
        data.columns = ["qualifier", "n"]
        chart = alt.Chart(data).mark_bar(color=color).encode(
            x=alt.X("n:Q", title="Ocorrências"),
            y=alt.Y("qualifier:N", sort="-x", title=""),
            tooltip=["qualifier", "n"],
        ).properties(title=label)
        st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 8 — PADRÕES TRANSVERSAIS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 8 · Padrões Transversais aos 5 Jogos")
st.markdown(
    f'<div class="info-box">Esta é a secção central: o que é <b>consistente</b> em todos os jogos? '
    f'O que varia com o adversário? Que tendências atravessam os 5 primeiros tempos?</div>',
    unsafe_allow_html=True
)

col_pt1, col_pt2 = st.columns(2)

with col_pt1:
    fase_jogo_bom = (df.groupby(["match_ref", "phase", "polarity"]).size()
                     .unstack(fill_value=0).reset_index())
    if "bom" not in fase_jogo_bom.columns: fase_jogo_bom["bom"] = 0
    if "mau" not in fase_jogo_bom.columns: fase_jogo_bom["mau"] = 0
    fase_jogo_bom["total"] = fase_jogo_bom["bom"] + fase_jogo_bom["mau"]
    fase_jogo_bom["pct_bom"] = (fase_jogo_bom["bom"] /
                                  fase_jogo_bom["total"].replace(0, 1) * 100)
    piv = fase_jogo_bom.pivot(index="match_ref", columns="phase", values="pct_bom").fillna(0)
    hm = piv.reset_index().melt(id_vars=["match_ref"], var_name="phase", value_name="pct")
    chart = alt.Chart(hm).mark_rect().encode(
        x=alt.X("phase:N", title="Fase"),
        y=alt.Y("match_ref:N", title="Jogo"),
        color=alt.Color("pct:Q", scale=alt.Scale(scheme="tealblues"), title="% BOM"),
        tooltip=["match_ref", "phase", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(title="% BOM por Fase × Jogo")
    st.altair_chart(chart, use_container_width=True)

with col_pt2:
    saldo = df.groupby("minute").apply(
        lambda x: (x["polarity"] == "bom").sum() - (x["polarity"] == "mau").sum()
    ).reset_index(name="saldo")
    saldo = saldo.sort_values("minute")
    chart = alt.Chart(saldo).mark_bar().encode(
        x=alt.X("minute:Q", title="Minuto"),
        y=alt.Y("saldo:Q", title="BOM − MAU"),
        color=alt.condition("datum.saldo >= 0", alt.value(GREEN), alt.value(RED))
    ).properties(title="Saldo BOM−MAU por minuto (acumulado todos os jogos)")
    st.altair_chart(chart, use_container_width=True)

st.markdown("#### Categorias presentes em todos (ou quase todos) os jogos")
cats_por_jogo = (df_cats.groupby(["categories", "match_ref"]).size()
                 .reset_index(name="n"))
cats_njogos = cats_por_jogo.groupby("categories")["match_ref"].nunique().reset_index()
cats_njogos.columns = ["categoria", "n_jogos"]
cats_com_freq = (df_cats.groupby("categories").size().reset_index(name="total"))
cats_njogos = cats_njogos.merge(cats_com_freq, left_on="categoria", right_on="categories")
cats_njogos = cats_njogos.sort_values(["n_jogos", "total"], ascending=False).head(12)

chart = alt.Chart(cats_njogos).mark_circle().encode(
    x=alt.X("total:Q", title="Total ocorrências"),
    y=alt.Y("n_jogos:Q", title="Nº de jogos em que aparece"),
    size=alt.Size("total:Q", scale=alt.Scale(range=[50, 800])),
    color=alt.Color("n_jogos:Q", scale=alt.Scale(scheme="blues")),
    tooltip=["categoria", "total", "n_jogos"],
).properties(title="Categorias mais transversais")

st.altair_chart(chart, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SECÇÃO 9 — EXPLORADOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 9 · Explorador de Momentos")

c_e1, c_e2, c_e3, c_e4 = st.columns(4)
pol_ex   = c_e1.selectbox("Polaridade", ["Todos", "bom", "mau"],
                            format_func=lambda x: POLARITY_LABEL.get(x, x))
fase_ex  = c_e2.selectbox("Fase", ["Todas"] + sorted(df["phase"].unique()))
jogo_ex  = c_e3.selectbox("Jogo", ["Todos"] + sorted(df["match_ref"].unique()))
prio_ex  = c_e4.selectbox("Prioridade", ["Todas"] + sorted(df["priority"].dropna().unique()))

df_ex = df.copy()
if pol_ex  != "Todos": df_ex = df_ex[df_ex["polarity"] == pol_ex]
if fase_ex != "Todas": df_ex = df_ex[df_ex["phase"] == fase_ex]
if jogo_ex != "Todos": df_ex = df_ex[df_ex["match_ref"] == jogo_ex]
if prio_ex != "Todas": df_ex = df_ex[df_ex["priority"] == prio_ex]

cols_show = ["analyst", "match_ref", "minute", "polarity", "phase", "categories",
             "qualifiers", "zone", "players_involved_names", "priority", "description"]
cols_ok = [c for c in cols_show if c in df_ex.columns]

st.dataframe(df_ex[cols_ok].reset_index(drop=True),
             use_container_width=True, height=340,
             column_config={
                 "polarity": st.column_config.TextColumn("Polar."),
                 "minute":   st.column_config.NumberColumn("Min.", format="%d'"),
                 "analyst":  st.column_config.TextColumn("Grupo"),
                 "match_ref": st.column_config.TextColumn("Jogo"),
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
