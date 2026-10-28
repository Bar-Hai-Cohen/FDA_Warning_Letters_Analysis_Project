"""
FDA Medical Warning Letters 2021–2026 – interactive dashboard (Streamlit).

Setup:
    pip install streamlit plotly pandas

Run from the folder that contains dashboard_data/ (or from the project folder – the notebook
writes it to data/dashboard_data/, which is found automatically):
    streamlit run dashboard.py
    # if "streamlit" is not on your PATH:
    python3 -m streamlit run dashboard.py

The data is created by section 9.1 of FDA_Final_fixed.ipynb. This app only reads those files;
no models are loaded or run. To use another folder: FDA_DASHBOARD_DATA=/path/to/dashboard_data
"""
import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FDA Warning Letters Dashboard", page_icon="💊", layout="wide")

st.markdown("""
<style>
.rtl, h1, h2, h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stAlert"] { direction: rtl; text-align: right; }
[data-testid="stMetric"] { direction: rtl; text-align: right; background: #f6f8fa;
                           border-radius: 8px; padding: 0.7rem 1rem; }
.evidence { direction: ltr; text-align: left; background: #f6f8fa; border-left: 4px solid #2E86AB;
            padding: 0.6rem 0.9rem; margin-bottom: 0.6rem; border-radius: 4px; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

HERE = Path(__file__).resolve().parent
TOTAL_COLOR = "#333333"
METHODS = {
    "K-Means (TF-IDF)":        ("kmeans_name", "K-Means"),
    "Semantic (SBERT + Ward)": ("semantic_name", "Semantic"),
    "BERTopic":                ("bertopic_name", None),
}


def find_data_dir():
    env = os.environ.get("FDA_DASHBOARD_DATA")
    candidates = ([Path(env)] if env else []) + [
        Path.cwd() / "dashboard_data", Path.cwd() / "data" / "dashboard_data",
        HERE / "dashboard_data", HERE / "data" / "dashboard_data",
    ]
    return next((c for c in candidates if (c / "letters.csv").exists()), None)


@st.cache_data
def load_data(data_dir):
    d = Path(data_dir)
    data = {
        "letters":         pd.read_csv(d / "letters.csv", parse_dates=["date", "month"]),
        "monthly":         pd.read_csv(d / "monthly_counts.csv", parse_dates=["month"]),
        "share":           pd.read_csv(d / "monthly_share.csv", parse_dates=["month"]),
        "cluster_monthly": pd.read_csv(d / "monthly_by_cluster.csv", parse_dates=["month"]),
        "forecast":        pd.read_csv(d / "prophet_forecast.csv", parse_dates=["ds"]),
        "backtest":        pd.read_csv(d / "prophet_backtest.csv"),
        "topics":          pd.read_csv(d / "bertopic_topics.csv"),
        "topics_time":     pd.read_csv(d / "bertopic_topics_over_time.csv", parse_dates=["Timestamp"]),
        "evidence":        pd.read_csv(d / "classification_evidence.csv"),
        "terms":           pd.read_csv(d / "distinctive_terms.csv"),
    }
    data["kpis"] = json.loads((d / "kpis.json").read_text(encoding="utf-8"))
    return data


def rtl(text):
    st.markdown(f'<div class="rtl">{text}</div>', unsafe_allow_html=True)


def show(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", height=height)
    st.plotly_chart(fig, use_container_width=True)


data_dir = find_data_dir()
if data_dir is None:
    st.error("לא נמצאה התיקייה dashboard_data/. יש להריץ קודם את סעיף 9.1 במחברת.")
    st.stop()

D = load_data(str(data_dir))
L, K = D["letters"], D["kpis"]
CAT_COLORS = K["category_colors"]
CATEGORIES = [c for c in CAT_COLORS if c in set(L["category"])]
SCORE_COLS = [c for c in L.columns if c.startswith("score: ")]


def color_of(name):
    return TOTAL_COLOR if name == "Total" else CAT_COLORS.get(name, "#777777")


# ─────────────────────────────── pages ───────────────────────────────
def page_overview():
    st.title("סקירה כללית")
    rtl(f"מכתבי אזהרה רפואיים של ה-FDA (תרופות וביולוגיה), לפי תאריך הוצאת המכתב. "
        f"הנתונים עד {K['last_date']}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("מכתבים", f"{K['n_letters']:,}")
    c2.metric("שנים", f"{K['year_min']}–{K['year_max']}")
    c3.metric("הסכמה K-Means ↔ סמנטי", f"{K['km_sem_agreement']:.0%}",
              help="אחוז המכתבים שנמצאים באשכול הסמנטי הנפוץ ביותר של אשכול ה-K-Means שלהם "
                   f"(Adjusted Rand = {K['km_sem_ari']:.2f})")
    c4.metric("הסכמת DeBERTa עם נושא ה-FDA", f"{K['deberta_agreement']:.0%}",
              help=f"על {K['n_reference']:,} מכתבים ששדה הנושא שלהם מתאים לאחת הקטגוריות")

    per_year = L.groupby("year").size().reset_index(name="letters")
    per_year["label"] = per_year["year"].astype(str) + per_year["year"].eq(K.get("partial_year")).map({True: "*", False: ""})
    left, right = st.columns(2)
    with left:
        show(px.bar(per_year, x="label", y="letters", text="letters", title="Letters per year",
                    labels={"label": "Year", "letters": "Letters"}, color_discrete_sequence=["#2E86AB"]))
    with right:
        cy = L.groupby(["year", "category"]).size().reset_index(name="letters")
        cy["share"] = cy["letters"] / cy.groupby("year")["letters"].transform("sum") * 100
        show(px.bar(cy, x="year", y="share", color="category", color_discrete_map=CAT_COLORS,
                    category_orders={"category": CATEGORIES}, hover_data={"letters": True, "share": ":.1f"},
                    title="DeBERTa category share per year (%)", labels={"share": "% of letters", "year": "Year"}))
    if K.get("partial_year"):
        st.caption(f"* {K['partial_year']} חלקית: נתונים עד {K['last_date']}.")


def page_monthly():
    st.title("מגמות חודשיות")
    M, S = D["monthly"], D["share"]
    rtl(f"ספירה חודשית לפי תאריך הוצאת המכתב, בלי החודש החלקי האחרון. ההחלקה היא LOWESS "
        f"(frac = {K['lowess_frac']}).")
    options = ["Total"] + CATEGORIES
    chosen = st.multiselect("סדרות", options, default=options)
    mode = st.radio("תצוגה", ["Raw + LOWESS", "LOWESS", "Raw", "Share % (LOWESS)"], horizontal=True)
    lo, hi = M["month"].min().to_pydatetime(), M["month"].max().to_pydatetime()
    start, end = st.slider("טווח תאריכים", min_value=lo, max_value=hi, value=(lo, hi), format="MMM YYYY")

    fig = go.Figure()
    if mode.startswith("Share"):
        s = S[S["month"].between(start, end)]
        for c in chosen:
            if c != "Total":
                fig.add_scatter(x=s["month"], y=s[f"lowess: {c}"], name=c, line=dict(color=color_of(c), width=3))
        fig.update_layout(title="Share of each category (% of that month's letters, LOWESS)",
                          yaxis=dict(range=[0, 100], title="% of letters"))
    else:
        m = M[M["month"].between(start, end)]
        for c in chosen:
            if "Raw" in mode:
                fig.add_scatter(x=m["month"], y=m[c], name=f"{c} (raw)", line=dict(color=color_of(c), width=1),
                                opacity=0.35 if "LOWESS" in mode else 1, showlegend=mode == "Raw")
            if "LOWESS" in mode:
                fig.add_scatter(x=m["month"], y=m[f"lowess: {c}"], name=c, line=dict(color=color_of(c), width=3))
        fig.update_layout(title="Letters per month", yaxis_title="Letters per month")
    show(fig, height=520)

    with st.expander("לפי אשכול K-Means (LOWESS)"):
        CM = D["cluster_monthly"]
        CM = CM[CM["month"].between(start, end)]
        clusters = [c for c in CM.columns if c != "month" and not c.startswith("lowess: ")]
        fig = go.Figure()
        for c in clusters:
            fig.add_scatter(x=CM["month"], y=CM[f"lowess: {c}"], name=c, line=dict(width=3))
        fig.update_layout(title="Letters per month by K-Means cluster (LOWESS)", yaxis_title="Letters per month")
        show(fig, height=480)


def page_clusters():
    st.title("אשכולות")
    method = st.radio("שיטה", list(METHODS), horizontal=True)
    col, terms_key = METHODS[method]
    order = L[col].value_counts().index.tolist()

    show(px.scatter(L, x="tsne_x", y="tsne_y", color=col, category_orders={col: order}, opacity=0.75,
                    hover_data={"company": True, "date": "|%Y-%m-%d", "subject": True,
                                "tsne_x": False, "tsne_y": False},
                    title=f"t-SNE map of the SBERT letter embeddings, colored by {method}",
                    labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2", col: "Cluster"}), height=620)

    left, right = st.columns(2)
    with left:
        sizes = L[col].value_counts().rename_axis("cluster").reset_index(name="letters")
        show(px.bar(sizes, x="letters", y="cluster", orientation="h", text="letters", title="Cluster sizes",
                    category_orders={"cluster": order}), height=max(300, 45 * len(order)))
    with right:
        def top_subject(s):
            vc = s.value_counts(normalize=True)
            return pd.Series({"top FDA subject": vc.index[0], "% of cluster": round(vc.iloc[0] * 100)})

        top = (L.assign(subject=L["subject"].fillna("(no subject)"))
               .groupby(col)["subject"].apply(top_subject).unstack().reindex(order))
        rtl("<b>נושא ה-FDA הנפוץ ביותר בכל אשכול</b>")
        st.dataframe(top, width="stretch")

    rtl("<b>מילים בולטות</b>")
    cluster = st.selectbox("אשכול", order)
    if terms_key:
        t = D["terms"].query("method == @terms_key and cluster_name == @cluster").sort_values("rank")
        show(px.bar(t[::-1], x="score", y="term", orientation="h",
                    title="Distinctive terms (mean TF-IDF in cluster − outside)"), height=420)
    else:
        row = D["topics"][D["topics"]["name"] == cluster]
        st.write(", ".join(row["top 8 keywords"]) if len(row) else "—")

    yc = L.groupby(["year", col]).size().reset_index(name="letters")
    show(px.bar(yc, x="year", y="letters", color=col, category_orders={col: order},
                title="Letters per year by cluster", labels={col: "Cluster", "year": "Year"}), height=500)


def page_topics():
    st.title("נושאים (BERTopic)")
    rtl(f"BERTopic מצא {K['n_bertopic_topics']} נושאים, על אותם embeddings של SBERT מסעיף 6. "
        "הטבלה מציגה את קבוצות מילות המפתח של כל נושא.")
    st.dataframe(D["topics"], hide_index=True, width="stretch")

    tt = D["topics_time"]
    names = D["topics"].sort_values("letters", ascending=False)["name"].tolist()
    chosen = st.multiselect("נושאים", names, default=names[:10])
    t = tt[tt["Name"].isin(chosen)]
    show(px.line(t, x="Timestamp", y="Frequency", color="Name", hover_data=["Words"], markers=True,
                 category_orders={"Name": names}, title="Topics over time (letters per month)",
                 labels={"Timestamp": "Month", "Frequency": "Letters"}), height=520)


def page_forecast():
    st.title("תחזית (Prophet)")
    st.warning("התחזית היא אינדיקציה בלבד: יש מעט נקודות חודשיות, ב-2025 יש שבירה מבנית, "
               "והחודשים האחרונים עשויים להיות חסרים בגלל עיכוב בפרסום. טבלת ה-backtest מראה אם "
               "Prophet טוב יותר מהבסיס הנאיבי.")
    F, B = D["forecast"], D["backtest"]
    name = st.selectbox("סדרה", F["series"].unique().tolist())
    f = F[F["series"] == name]
    hist, fut = f[f["type"] == "fit"], f[f["type"] == "forecast"]
    joined = pd.concat([hist.tail(1), fut])

    fig = go.Figure()
    fig.add_bar(x=hist["ds"], y=hist["actual"], name="Actual", marker_color="#BBBBBB")
    fig.add_scatter(x=hist["ds"], y=hist["yhat"], name="Prophet fit", line=dict(color="#2E86AB", width=2))
    fig.add_scatter(x=pd.concat([joined["ds"], joined["ds"][::-1]]),
                    y=pd.concat([joined["yhat_upper"], joined["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(232,72,85,0.2)", line=dict(width=0), name="80% interval")
    fig.add_scatter(x=joined["ds"], y=joined["yhat"], name=f"Forecast (+{K['horizon']} months)",
                    mode="lines+markers", line=dict(color=color_of(name) if name != "Total" else "#E84855", width=3))
    fig.add_vline(x=hist["ds"].max(), line_dash="dash", line_color="black")
    fig.update_layout(title=f"{name}: actual, fit and forecast", yaxis_title="Letters per month")
    show(fig, height=520)

    rtl("<b>Backtest</b> – אימון בלי החודשים האחרונים, חיזוי שלהם, ו-MAE מול בסיס נאיבי (ממוצע 6 החודשים שלפני).")
    st.dataframe(B.style.apply(lambda r: ["background-color: #fff3cd" if r["series"] == name else "" for _ in r],
                               axis=1), hide_index=True, width="stretch")


def page_explorer():
    st.title("סייר המכתבים")
    f1, f2, f3, f4 = st.columns(4)
    years = f1.multiselect("שנה", sorted(L["year"].unique()))
    cats = f2.multiselect("קטגוריה", CATEGORIES)
    clusters = f3.multiselect("אשכול K-Means", sorted(L["kmeans_name"].unique()))
    query = f4.text_input("חיפוש חברה")

    sel = L
    if years:
        sel = sel[sel["year"].isin(years)]
    if cats:
        sel = sel[sel["category"].isin(cats)]
    if clusters:
        sel = sel[sel["kmeans_name"].isin(clusters)]
    if query:
        sel = sel[sel["company"].fillna("").str.contains(query, case=False, regex=False)]
    sel = sel.sort_values("date", ascending=False)

    st.caption(f"{len(sel):,} מכתבים")
    st.dataframe(sel[["date", "company", "category", "kmeans_name", "bertopic_name", "subject"]],
                 hide_index=True, width="stretch", height=320,
                 column_config={"date": st.column_config.DateColumn("date", format="YYYY-MM-DD")})
    if sel.empty:
        st.info("אין מכתבים שמתאימים לסינון.")
        return

    labels = (sel["company"].fillna("?") + " — " + sel["date"].dt.strftime("%Y-%m-%d")).tolist()
    pick = st.selectbox("בחר מכתב", range(len(sel)), format_func=lambda i: labels[i])
    row = sel.iloc[pick]

    left, right = st.columns([1, 1])
    with left:
        st.subheader(str(row["company"]))
        meta = {
            "Date": f"{row['date']:%Y-%m-%d}",
            "Issuing office": row["office"],
            "FDA subject": row["subject"],
            "DeBERTa category": row["category"],
            "Reference (from FDA subject)": row["reference_topic"] if pd.notna(row["reference_topic"]) else "—",
            "K-Means cluster": row["kmeans_name"],
            "Semantic cluster": row["semantic_name"],
            "BERTopic topic": row["bertopic_name"],
            "Relevant chunks": int(row["relevant_chunks"]),
        }
        st.table(pd.Series(meta, name="").astype(str))
        st.link_button("פתח את המכתב באתר ה-FDA", row["url"])
    with right:
        scores = pd.DataFrame({"topic": [c.removeprefix("score: ") for c in SCORE_COLS],
                               "score": [row[c] for c in SCORE_COLS]})
        if scores["score"].notna().any():
            fig = px.bar(scores, x="score", y="topic", orientation="h", color="topic",
                         color_discrete_map=CAT_COLORS, range_x=[0, 1],
                         title="Mean DeBERTa score over the relevant chunks")
            fig.update_layout(showlegend=False)
            show(fig, height=320)
        else:
            st.info("למכתב הזה אין קטעים רלוונטיים, ולכן אין ציונים.")

    ev = D["evidence"][D["evidence"]["url"] == row["url"]].sort_values("rank")
    rtl("<b>פסקאות ראיה</b> – הקטעים עם הציון הגבוה ביותר לקטגוריה שנבחרה")
    if ev.empty:
        st.info("אין פסקאות ראיה (המכתב לא סווג).")
    for _, e in ev.iterrows():
        st.markdown(f'<div class="evidence"><b>#{e["rank"]} · {html.escape(str(e["topic"]))} · '
                    f'score {e["score"]:.2f}</b><br>{html.escape(str(e["text"]))}</div>', unsafe_allow_html=True)


PAGES = {
    "סקירה כללית": page_overview,
    "מגמות חודשיות": page_monthly,
    "אשכולות": page_clusters,
    "נושאים (BERTopic)": page_topics,
    "תחזית": page_forecast,
    "סייר המכתבים": page_explorer,
}
st.sidebar.title("FDA Warning Letters")
page = st.sidebar.radio("ניווט", list(PAGES))
st.sidebar.caption(f"{K['n_letters']:,} מכתבים · עד {K['last_date']}")
st.sidebar.caption(f"נתונים: {data_dir}")
PAGES[page]()
