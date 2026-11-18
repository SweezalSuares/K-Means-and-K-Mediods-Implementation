# Customer Segmentation: K-Means vs K-Medoids

A Streamlit app built from the lab notebook `notebooks/SweezalSuares_547_lab7.ipynb`.
It segments ~9,000 credit-card customers with **K-Means** and **K-Medoids**, picks the number of clusters,
evaluates both algorithms and lets you assign a new customer to a segment.

## What the app shows

| Tab | Content |
|---|---|
| 1 · Data & preprocessing | Dataset description, missing values, skewness/outliers, scaling, feature histograms |
| 2 · K-Means | Elbow method, silhouette analysis, PCA cluster plot with centroids, centroid tables and heatmap |
| 3 · K-Medoids | Cost curve, silhouette analysis, cluster plot with medoids, medoid customers (real `CUST_ID`s) |
| 4 · Comparison & outliers | Silhouette, Davies-Bouldin and Calinski-Harabasz for both; metrics across k; centroids vs medoids; outlier robustness test |
| 5 · Interpretation | Cluster profiles, feature box plots, justification of k, conclusion |
| 6 · Segment a customer | Enter a customer's behaviour and get the nearest K-Means and K-Medoids segment |

The sidebar lets you change **k**, the **random seed**, the **extreme-customer threshold**, and **upload your own CSV**
(numeric columns; a `CUST_ID` column is ignored).

## Pipeline (same as the notebook)

1. Remove duplicate rows and the `CUST_ID` identifier (the data has no class label).
2. Fill missing values (`MINIMUM_PAYMENTS`, `CREDIT_LIMIT`) with the **median**.
3. Standardise all 17 features (mean 0, std 1).
4. K-Means (`n_init=20`) and K-Medoids (`alternate` update, k-medoids++ init), random seed 42.
5. Choose k with the elbow method and silhouette score; evaluate with three internal metrics.

## Key results (k = 3, seed 42)

| Metric | K-Means | K-Medoids |
|---|---|---|
| Silhouette (higher is better) | 0.251 | 0.172 |
| Davies-Bouldin (lower is better) | 1.597 | 1.668 |
| Calinski-Harabasz (higher is better) | 1605 | 1468 |

K-Means is the better fit for this dataset. All silhouette scores are low (≤ 0.25), so the clusters overlap and the
structure is only moderate. The three metrics are all Euclidean-to-mean based, which favours K-Means.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The first load takes about 30 seconds because it fits both algorithms for k = 2 to 10;
results are cached afterwards.

### K-Medoids backend

* **Default (`requirements.txt`)**: the app uses a **built-in** K-Medoids implementation, so there is nothing to compile.
  It uses the same algorithm family, but its clusters can differ slightly from the notebook's.
* **Exact notebook numbers**: `pip install scikit-learn-extra`. The app detects it automatically and the sidebar shows
  *K-Medoids backend: scikit-learn-extra*. This package needs a C compiler on some Python versions; if the install
  fails, the built-in version keeps working.

To run the notebook as well: `pip install -r requirements-notebook.txt`, then `jupyter notebook`.

## Project structure

```
segmentation-app/
├── app.py                  Streamlit interface
├── clustering.py           Preprocessing, K-Means, K-Medoids, metrics (no Streamlit code)
├── data/CC_GENERAL.csv     Kaggle credit-card dataset (8,950 customers)
├── notebooks/              Original lab notebook
├── tests/test_app.py       Smoke tests (pytest -q)
├── .streamlit/config.toml  Theme and server settings
├── requirements.txt        App dependencies
└── requirements-notebook.txt  Extra packages for the notebook / exact K-Medoids
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On https://share.streamlit.io choose **New app**, select the repo and set the main file to `app.py`.
3. Use `requirements.txt` as is (it deliberately leaves out `scikit-learn-extra`).

## Notes

* The built-in K-Medoids builds an 8,950 × 8,950 distance matrix (about 320 MB). If the host has little memory
  (e.g. 1 GB), install `scikit-learn-extra` or upload a smaller CSV.
* The conclusion text in tab 5 reflects the default settings (k = 3, seed 42); tables and plots update with your settings.
* In the notebook, add `from sklearn.impute import SimpleImputer` to the import cell so it also runs after
  *Restart & Run All*.

## Dataset

Credit Card Dataset for Clustering (Kaggle): usage behaviour of ~9,000 active credit-card holders over 6 months,
17 behavioural features per customer.
