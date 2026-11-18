"""Credit-card customer segmentation: K-Means vs K-Medoids (Streamlit app).

Run with:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

import clustering as cl

st.set_page_config(page_title="Customer Segmentation: K-Means vs K-Medoids",
                   page_icon="💳", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "CC_GENERAL.csv"
SEED_DEFAULT = 42
C_KM, C_MED = "#1f77b4", "#ff7f0e"


# ------------------------------------------------------------------------ caching
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Preprocessing data...")
def prepare(df: pd.DataFrame):
    return cl.preprocess(df)


@st.cache_resource(show_spinner="Building distance matrix (built-in K-Medoids only)...")
def get_dist(_Xs, key):
    return cl.distance_matrix(_Xs)


@st.cache_data(show_spinner="Running K-Means for k = 2 to 10...")
def sweep_kmeans(_Xs, key, seed):
    rows = []
    for k in range(cl.K_MIN, cl.K_MAX + 1):
        labels, _, inertia = cl.fit_kmeans(_Xs, k, seed, n_init=10)
        e = cl.evaluate(_Xs, labels)
        rows.append({"k": k, "WCSS": inertia, **e})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running K-Medoids for k = 2 to 10 (this takes a little while)...")
def sweep_kmedoids(_Xs, key, seed):
    D = (lambda: get_dist(_Xs, key))
    rows = []
    for k in range(cl.K_MIN, cl.K_MAX + 1):
        labels, _, cost = cl.fit_kmedoids_auto(_Xs, D, k, seed, max_iter=200)
        e = cl.evaluate(_Xs, labels)
        rows.append({"k": k, "Total cost": cost, **e})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Fitting K-Means...")
def run_kmeans(_Xs, key, k, seed):
    return cl.fit_kmeans(_Xs, k, seed, n_init=20)


@st.cache_data(show_spinner="Fitting K-Medoids...")
def run_kmedoids(_Xs, key, k, seed):
    return cl.fit_kmedoids_auto(_Xs, (lambda: get_dist(_Xs, key)), k, seed, max_iter=300)


@st.cache_data(show_spinner="Running the outlier robustness test...")
def robustness(_Xs, key, k, seed, z):
    ext = cl.extreme_mask(_Xs, z)
    keep = np.flatnonzero(~ext)
    km_l, km_c, _ = cl.fit_kmeans(_Xs[keep], k, seed, n_init=20)
    md_l, md_i, _ = cl.fit_kmedoids_auto(_Xs, (lambda: get_dist(_Xs, key)), k, seed, rows=keep, max_iter=300)
    md_c = _Xs[keep][md_i]
    return ext, km_l, km_c, md_l, md_c, _Xs[keep]


@st.cache_data(show_spinner=False)
def run_pca(_Xs, key):
    p = PCA(n_components=2, random_state=0).fit(_Xs)
    return p.transform(_Xs), p.components_, p.mean_, float(p.explained_variance_ratio_.sum())


# --------------------------------------------------------------------------- plots
def scatter_clusters(Z, labels, centres_2d, names, title, marker_name):
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for c in range(len(names)):
        m = labels == c
        fig.add_trace(go.Scattergl(x=Z[m, 0], y=Z[m, 1], mode="markers", name=names[c],
                                   marker=dict(size=4, opacity=0.5, color=palette[c % len(palette)])))
    fig.add_trace(go.Scatter(x=centres_2d[:, 0], y=centres_2d[:, 1], mode="markers+text",
                             text=[str(i) for i in range(len(names))], textfont=dict(color="white"),
                             name=marker_name,
                             marker=dict(symbol="x", size=18, color="black", line=dict(color="white", width=2))))
    lo, hi = np.percentile(Z, 0.5, axis=0), np.percentile(Z, 99.5, axis=0)
    fig.update_layout(title=title, xaxis=dict(title="PC1", range=[lo[0], hi[0]]),
                      yaxis=dict(title="PC2", range=[lo[1], hi[1]]), height=520,
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50))
    return fig


def line_pair(x, y1, y2, t1, t2, yl1, yl2, mark1=None, mark2=None, color=C_KM):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(t1, t2))
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines+markers", line=dict(color=color), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines+markers", line=dict(color="green"), showlegend=False), 1, 2)
    if mark1:
        fig.add_vline(x=mark1, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text=f"elbow k={mark1}")
    if mark2:
        fig.add_vline(x=mark2, line_dash="dash", line_color="red", row=1, col=2,
                      annotation_text=f"best k={mark2}")
    fig.update_xaxes(title_text="Number of clusters (k)")
    fig.update_yaxes(title_text=yl1, row=1, col=1)
    fig.update_yaxes(title_text=yl2, row=1, col=2)
    fig.update_layout(height=400, margin=dict(t=50))
    return fig


def heat(centres_z, features, title):
    df = pd.DataFrame(centres_z, columns=features).T
    df.columns = [f"Cluster {i}" for i in range(df.shape[1])]
    fig = px.imshow(df, text_auto=".2f", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    aspect="auto", title=title)
    fig.update_layout(height=620, margin=dict(t=50))
    return fig


# ---------------------------------------------------------------------------- sidebar
st.sidebar.title("💳 Settings")
uploaded = st.sidebar.file_uploader("Use your own CSV (optional)", type="csv",
                                    help="Needs numeric columns. A CUST_ID column is ignored if present.")
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        data_name = uploaded.name
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read that file: {e}")
        st.stop()
else:
    if not DATA_PATH.exists():
        st.error(f"Default dataset not found at {DATA_PATH}. Upload a CSV in the sidebar.")
        st.stop()
    raw_df = load_default()
    data_name = "CC_GENERAL.csv (Kaggle credit-card dataset)"

try:
    P = prepare(raw_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

data_key = f"{data_name}-{P.X_scaled.shape}"
Xs = P.X_scaled

K = st.sidebar.slider("Number of clusters (k)", cl.K_MIN, cl.K_MAX, 3,
                      help="The notebook uses k = 3 (best K-Means silhouette).")
seed = st.sidebar.number_input("Random seed", 0, 9999, SEED_DEFAULT, step=1)
z_thr = st.sidebar.slider("Extreme-customer threshold (|z| >)", 3.0, 8.0, 5.0, 0.5,
                          help="Used in the outlier analysis.")
st.sidebar.caption(f"K-Medoids backend: **{cl.KMEDOIDS_BACKEND}**")
if not cl.HAS_SKLEARN_EXTRA:
    st.sidebar.info("Using the built-in K-Medoids. Install `scikit-learn-extra` "
                    "(see README) to reproduce the notebook's K-Medoids numbers exactly.")

# main fits for the chosen k
km_labels, km_centres, km_inertia = run_kmeans(Xs, data_key, K, int(seed))
md_labels, md_idx, md_cost = run_kmedoids(Xs, data_key, K, int(seed))
md_centres = Xs[md_idx]
Z, pca_comp, pca_mean, pca_var = run_pca(Xs, data_key)
km_names = cl.name_clusters(km_centres, P.features)
md_names = cl.name_clusters(md_centres, P.features)

# ------------------------------------------------------------------------------ header
st.title("Customer Segmentation: K-Means vs K-Medoids")
st.caption(f"Dataset: **{data_name}** · {Xs.shape[0]:,} customers · {Xs.shape[1]} numeric features · "
           f"k = {K} · seed = {int(seed)}")

tabs = st.tabs(["1 · Data & preprocessing", "2 · K-Means", "3 · K-Medoids",
                "4 · Comparison & outliers", "5 · Interpretation", "6 · Segment a customer"])

# ================================================================== 1. data
with tabs[0]:
    st.subheader("Dataset description")
    st.write(
        "Usage behaviour of credit-card holders (balance, purchases, cash advances, payments, "
        "credit limit...). There is **no class label**, so the data is naturally suited to "
        "clustering. `CUST_ID` is only an identifier and is removed.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{P.n_rows_in:,}")
    c2.metric("Duplicate rows removed", P.n_duplicates)
    c3.metric("Numeric features", len(P.features))
    c4.metric("Missing values (before imputation)", int(P.missing.sum()))

    st.markdown("**Raw data preview**")
    st.dataframe(raw_df.head(100), width="stretch", height=250)

    a, b = st.columns(2)
    with a:
        st.markdown("**Missing values** (filled with the column **median**)")
        miss = P.missing[P.missing > 0]
        st.dataframe(miss.rename("missing").to_frame() if len(miss) else
                     pd.DataFrame({"missing": ["none"]}), width="stretch")
    with b:
        st.markdown("**Skewness and outliers** (IQR rule)")
        st.dataframe(cl.skew_outlier_table(P.X), width="stretch", height=260)

    st.markdown("**Summary statistics (after imputation)**")
    st.dataframe(P.X.describe().T.round(2), width="stretch")

    st.info("**Why scale?** K-Means and K-Medoids are distance based. Without standardisation, "
            "`BALANCE` or `PURCHASES` (in the thousands) would dominate the 0-1 `*_FREQUENCY` "
            "columns. Every feature is standardised to mean 0 and standard deviation 1.")

    f = st.selectbox("Inspect a feature", P.features,
                     index=P.features.index("PURCHASES") if "PURCHASES" in P.features else 0)
    h1, h2 = st.columns(2)
    h1.plotly_chart(px.histogram(P.X, x=f, nbins=60, title=f"{f} (raw)"), width="stretch")
    h2.plotly_chart(px.histogram(pd.DataFrame({f: Xs[:, P.features.index(f)]}), x=f, nbins=60,
                                 title=f"{f} (standardised)"), width="stretch")

# ================================================================== 2. K-Means
with tabs[1]:
    st.subheader("Choosing k: Elbow Method and Silhouette Score")
    sk = sweep_kmeans(Xs, data_key, int(seed))
    k_el = cl.elbow_k(sk["k"], sk["WCSS"])
    k_si = int(sk.loc[sk["Silhouette"].idxmax(), "k"])
    st.plotly_chart(line_pair(sk["k"], sk["WCSS"], sk["Silhouette"], "K-Means: Elbow Method",
                              "K-Means: Silhouette Analysis", "WCSS (inertia)", "Silhouette score",
                              k_el, k_si), width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Elbow point (automatic)", k_el)
    m2.metric("Best silhouette", f"k = {k_si}", f"{sk['Silhouette'].max():.3f}", delta_color="off")
    m3.metric("Selected k", K)
    st.caption("The WCSS curve bends gradually (no sharp elbow), so the Silhouette Score is used to "
               "decide between the candidate values.")
    with st.expander("Table of values"):
        st.dataframe(sk.set_index("k").round(4), width="stretch")

    st.subheader(f"K-Means clusters (k = {K}) with centroids")
    st.caption(f"PCA projection for display only (PC1 + PC2 explain {pca_var*100:.1f}% of the variance); "
               "clustering is done in the full standardised space. Axes are clipped to the central 99%.")
    st.plotly_chart(scatter_clusters(Z, km_labels, (km_centres - pca_mean) @ pca_comp.T,
                                     [f"{i}: {n}" for i, n in enumerate(km_names)],
                                     f"K-Means (k={K})", "Centroids"), width="stretch")

    st.subheader("Cluster centroids")
    cen = pd.DataFrame(P.scaler.inverse_transform(km_centres), columns=P.features).round(2)
    cen.index = [f"{i}: {n}" for i, n in enumerate(km_names)]
    cen["Size"] = np.bincount(km_labels, minlength=K)
    st.dataframe(cen.T, width="stretch", height=640)
    st.plotly_chart(heat(km_centres, P.features, "K-Means centroids (standardised units)"),
                    width="stretch")

# ================================================================== 3. K-Medoids
with tabs[2]:
    st.subheader("Choosing k: cost curve and Silhouette Score")
    sm = sweep_kmedoids(Xs, data_key, int(seed))
    k_si_m = int(sm.loc[sm["Silhouette"].idxmax(), "k"])
    st.plotly_chart(line_pair(sm["k"], sm["Total cost"], sm["Silhouette"], "K-Medoids: cost curve",
                              "K-Medoids: Silhouette Analysis", "Total cost (sum of distances)",
                              "Silhouette score", None, k_si_m, color="darkorange"),
                    width="stretch")
    st.caption("The cost curve has no clear elbow. The Silhouette is highest at the smallest k, which "
               "is too coarse to be useful, so the same k as K-Means is used for a fair comparison.")
    with st.expander("Table of values"):
        st.dataframe(sm.set_index("k").round(4), width="stretch")

    st.subheader(f"K-Medoids clusters (k = {K}) with medoids")
    st.plotly_chart(scatter_clusters(Z, md_labels, (md_centres - pca_mean) @ pca_comp.T,
                                     [f"{i}: {n}" for i, n in enumerate(md_names)],
                                     f"K-Medoids (k={K})", "Medoids"), width="stretch")

    st.subheader("Cluster medoids (real customers)")
    med = P.X.iloc[md_idx].round(2).copy()
    med.index = [f"{i}: {n}" for i, n in enumerate(md_names)]
    med["Size"] = np.bincount(md_labels, minlength=K)
    med["CUST_ID"] = P.ids.iloc[md_idx].values
    st.dataframe(med.T.astype(str), width="stretch", height=640)
    st.plotly_chart(heat(md_centres, P.features, "K-Medoids medoids (standardised units)"),
                    width="stretch")

# ================================================================== 4. comparison
with tabs[3]:
    st.subheader(f"Evaluation metrics at k = {K}")
    e_km, e_md = cl.evaluate(Xs, km_labels), cl.evaluate(Xs, md_labels)
    comp = pd.DataFrame({"K-Means": e_km, "K-Medoids": e_md}).T
    comp["Cost (WCSS / sum of distances)"] = [km_inertia, md_cost]
    comp["Cluster sizes"] = [str(np.bincount(km_labels, minlength=K).tolist()),
                             str(np.bincount(md_labels, minlength=K).tolist())]
    st.dataframe(comp.style.format({"Silhouette": "{:.4f}", "Davies-Bouldin": "{:.4f}",
                                    "Calinski-Harabasz": "{:.1f}",
                                    "Cost (WCSS / sum of distances)": "{:,.0f}"}),
                 width="stretch")
    w = [("Silhouette ↑", "K-Means" if e_km["Silhouette"] >= e_md["Silhouette"] else "K-Medoids"),
         ("Davies-Bouldin ↓", "K-Means" if e_km["Davies-Bouldin"] <= e_md["Davies-Bouldin"] else "K-Medoids"),
         ("Calinski-Harabasz ↑", "K-Means" if e_km["Calinski-Harabasz"] >= e_md["Calinski-Harabasz"] else "K-Medoids")]
    st.write("**Better on each metric:** " + " · ".join(f"{a}: **{b}**" for a, b in w))
    st.caption("The cost columns are not comparable (squared vs plain distance). All three metrics are "
               "built on Euclidean distance to means, which naturally favours K-Means.")

    st.subheader("Metrics across all k")
    mk = sk.assign(Algorithm="K-Means")[["k", "Algorithm", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]]
    mm = sm.assign(Algorithm="K-Medoids")[["k", "Algorithm", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]]
    allk = pd.concat([mk, mm])
    cols = st.columns(3)
    for col, met in zip(cols, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]):
        fig = px.line(allk, x="k", y=met, color="Algorithm", markers=True, title=met,
                      color_discrete_map={"K-Means": C_KM, "K-Medoids": C_MED})
        fig.add_vline(x=K, line_dash="dot", line_color="grey")
        fig.update_layout(height=340, legend=dict(orientation="h", y=-0.25), margin=dict(t=40))
        col.plotly_chart(fig, width="stretch")

    st.subheader("Centroids vs medoids")
    i, j, dist = cl.match_centres(km_centres, md_centres)
    match = pd.DataFrame({"K-Means cluster": [f"{a}: {km_names[a]}" for a in i],
                          "Matched K-Medoids cluster": [f"{b}: {md_names[b]}" for b in j],
                          "Distance (std. units)": dist.round(3),
                          "K-Means size": np.bincount(km_labels, minlength=K)[i],
                          "K-Medoids size": np.bincount(md_labels, minlength=K)[j]})
    st.dataframe(match, width="stretch", hide_index=True)
    mapping = dict(zip(j, i))
    aligned = np.array([mapping[l] for l in md_labels])
    nearest = [np.linalg.norm(Xs - c, axis=1).min() for c in km_centres]
    q1, q2, q3 = st.columns(3)
    q1.metric("Adjusted Rand Index", f"{adjusted_rand_score(km_labels, md_labels):.3f}",
              help="1 = identical partitions, 0 = no better than chance")
    q2.metric("Same cluster in both", f"{(aligned == km_labels).mean()*100:.1f}%")
    q3.metric("Centroid → nearest real customer", f"{np.mean(nearest):.2f} (avg)",
              help="Medoids are real customers, so for them this distance is 0.")
    side = pd.concat({"K-Means centroid": pd.DataFrame(P.scaler.inverse_transform(km_centres[i]),
                                                       columns=P.features),
                      "K-Medoids medoid": P.X.iloc[md_idx[j]].reset_index(drop=True)}, axis=1)
    with st.expander("Matched centres side by side (original units)"):
        st.dataframe(side.T.round(1), width="stretch", height=640)

    st.subheader("Influence of outliers")
    ext, km_in_l, km_in_c, md_in_l, md_in_c, Xin = robustness(Xs, data_key, K, int(seed), float(z_thr))
    o1, o2 = st.columns(2)
    o1.metric(f"Extreme customers (|z| > {z_thr:g})", f"{int(ext.sum()):,}", f"{ext.mean()*100:.1f}% of all",
              delta_color="off")
    dist_df = pd.DataFrame({"K-Means: all": np.bincount(km_labels, minlength=K),
                            "K-Means: extreme": np.bincount(km_labels[ext], minlength=K),
                            "K-Medoids: all": np.bincount(md_labels, minlength=K),
                            "K-Medoids: extreme": np.bincount(md_labels[ext], minlength=K)})
    dist_df.index.name = "Cluster"
    o2.dataframe(dist_df, width="stretch")

    _, _, s_km = cl.match_centres(km_centres, km_in_c)
    _, _, s_md = cl.match_centres(md_centres, md_in_c)
    shift = pd.DataFrame({"K-Means centre shift": s_km, "K-Medoids centre shift": s_md}).round(3)
    shift.index = [f"Cluster {i}" for i in range(len(shift))]
    avg = shift.mean().round(3)
    shift.loc["Average"] = avg
    s1, s2 = st.columns([1, 1])
    s1.markdown("**Robustness test:** remove the extreme customers, refit both algorithms and measure how "
                "far each centre moves (standardised units). A smaller shift means a more stable method.")
    s1.dataframe(shift, width="stretch")
    fig = px.bar(shift.iloc[:-1], barmode="group", title="Centre shift after removing extreme customers",
                 color_discrete_map={"K-Means centre shift": C_KM, "K-Medoids centre shift": C_MED})
    fig.update_layout(height=320, yaxis_title="Distance moved", xaxis_title="Cluster")
    s2.plotly_chart(fig, width="stretch")
    more = "K-Means" if avg.iloc[0] > avg.iloc[1] else "K-Medoids"
    st.warning(f"In this run the **{more}** centres moved more on average "
               f"(K-Means {avg.iloc[0]:.2f} vs K-Medoids {avg.iloc[1]:.2f}). The usual claim that medoids are "
               "always more robust does not automatically hold when clusters overlap heavily and the "
               "K-Medoids cost surface is flat. Try other seeds to see how stable this result is.")

# ================================================================== 5. interpretation
with tabs[4]:
    st.subheader("Cluster characteristics")
    prof = P.X.copy()
    prof["KMeans"], prof["KMedoids"] = km_labels, md_labels
    key = [c for c in cl.KEY_FEATURES if c in P.features] or P.features[:10]
    a, b = st.columns(2)
    with a:
        st.markdown("**K-Means** (cluster mean)")
        t = prof.groupby("KMeans")[key].mean().round(2)
        t.index = [f"{x}: {km_names[x]}" for x in t.index]
        t["Size"] = np.bincount(km_labels, minlength=K)
        st.dataframe(t, width="stretch")
    with b:
        st.markdown("**K-Medoids** (cluster median)")
        t = prof.groupby("KMedoids")[key].median().round(2)
        t.index = [f"{x}: {md_names[x]}" for x in t.index]
        t["Size"] = np.bincount(md_labels, minlength=K)
        st.dataframe(t, width="stretch")
    st.caption("Segment names are generated by simple rules from the cluster centres "
               "(e.g. high cash advance and cash-advance frequency → *Cash-advance users*).")

    ft = st.selectbox("Compare a feature across clusters", key)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("K-Means", "K-Medoids"), shared_yaxes=True)
    for col, (lab, nm) in enumerate([(km_labels, km_names), (md_labels, md_names)], start=1):
        fig.add_trace(go.Box(y=np.log1p(P.X[ft].clip(lower=0)), x=[f"{l}: {nm[l]}" for l in lab],
                             showlegend=False, marker_color=C_KM if col == 1 else C_MED), 1, col)
    fig.update_yaxes(title_text=f"log(1 + {ft})", row=1, col=1)
    fig.update_layout(height=420, margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Why k = 3?")
    st.markdown(
        "- **Silhouette (K-Means)** peaks at k = 3 (≈ 0.25), clearly above k = 4-6.\n"
        "- **Elbow:** the WCSS bends gradually around k = 3-4, with no sharp elbow, so the silhouette decides.\n"
        "- **K-Medoids:** no sharp elbow; k = 3 is used for a like-for-like comparison.\n"
        "- **Interpretability:** three segments (spenders, cash-advance users, low/moderate users) match "
        "well-known credit-card behaviour and each suggests an action.\n"
        "- **Limitation:** silhouette scores are low (≤ 0.25), so clusters overlap and the structure is moderate.")

    st.subheader("Conclusion")
    st.success(
        "**K-Means is the more suitable algorithm for this dataset** (at k = 3 it scores better on Silhouette, "
        "Davies-Bouldin and Calinski-Harabasz). It isolates the small but valuable high-spender and "
        "cash-advance groups and is cheaper on larger data. K-Medoids remains useful for interpretability: "
        "its medoids are real customers and describe the *typical* customer rather than the average.")
    st.caption("This text reflects the notebook results at the default settings (k = 3, seed 42). "
               "The tables and plots above update with your settings.")

# ================================================================== 6. predict
with tabs[5]:
    st.subheader("Which segment does a customer belong to?")
    st.write("Enter a customer's behaviour. Each value starts at the dataset **median**. The customer is "
             "assigned to the nearest K-Means centroid and the nearest K-Medoids medoid (in standardised space).")
    form_feats = [c for c in cl.KEY_FEATURES if c in P.features] or P.features[:10]
    med_all, hi_all = P.X.median(), P.X.quantile(0.99)
    vals = {}
    with st.form("predict"):
        cols = st.columns(3)
        for n, f in enumerate(form_feats):
            top = float(max(hi_all[f] * 1.5, P.X[f].max() if P.X[f].max() <= 1 else hi_all[f] * 1.5))
            vals[f] = cols[n % 3].number_input(f, min_value=0.0, value=float(round(med_all[f], 2)),
                                               step=max(top / 100, 0.01), format="%.2f")
        go_btn = st.form_submit_button("Assign segment", type="primary")
    if go_btn:
        row = med_all.copy()
        for f, v in vals.items():
            row[f] = v
        z = P.scaler.transform(pd.DataFrame([row[P.features]]))[0]
        d_km = np.linalg.norm(km_centres - z, axis=1)
        d_md = np.linalg.norm(md_centres - z, axis=1)
        r1, r2 = st.columns(2)
        r1.metric("K-Means segment", f"{d_km.argmin()}: {km_names[d_km.argmin()]}")
        r2.metric("K-Medoids segment", f"{d_md.argmin()}: {md_names[d_md.argmin()]}")
        dd = pd.DataFrame({"K-Means distance": d_km.round(3), "K-Medoids distance": d_md.round(3)})
        dd.index = [f"Cluster {i}" for i in range(K)]
        st.dataframe(dd, width="stretch")
        st.caption("Features not shown in the form are held at their dataset median.")

st.divider()
st.caption("Built from the lab notebook `notebooks/SweezalSuares_547_lab7.ipynb`.")
