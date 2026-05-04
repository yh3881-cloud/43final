# Spatial-Temporal Discovery of Undervalued Airbnb Listings in NYC

## Project Objective
This project discovers **undervalued Airbnb listings** in NYC by combining spatial and temporal signals, then using machine learning to estimate each listing's theoretical market price.

A listing is treated as a good deal when its observed `effective_price` is meaningfully below its model-predicted price **while maintaining strong ratings and sufficient review evidence**.

## Data Sources
1. Inside Airbnb NYC detailed listings  
   - https://insideairbnb.com/get-the-data/
2. Inside Airbnb NYC detailed calendar  
   - URL pattern: `https://data.insideairbnb.com/united-states/ny/new-york-city/YYYY-MM-DD/data/calendar.csv.gz`
3. Inside Airbnb NYC detailed reviews  
   - URL pattern: `https://data.insideairbnb.com/united-states/ny/new-york-city/YYYY-MM-DD/data/reviews.csv.gz`
4. MTA Subway Stations (New York State Open Data)  
   - Dataset page: https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f  
   - CSV API: https://data.ny.gov/resource/39hk-dx4f.csv?$limit=50000
5. NYPD Complaint Data (NYC Open Data)
   - Historic dataset: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
   - Current Year To Date dataset: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243

## Repository Structure
```text
43final/
├── data/
│   └── raw and processed project data
├── notebooks/
│   └── EDA and exploratory scripts
├── outputs/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── undervalued_candidates.csv
│   └── undervalued_cluster_summary.csv
├── src/
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_data_quality_report.py
│   ├── 04_price_prediction.py
│   ├── 05_undervalued_clustering.py
│   └── app_airbnb_dashboard.py
├── 43_project4_final_report.pdf
├── README.md
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Pipeline

Run the full reproducible pipeline from the project root.

```bash
# 1. Download raw source data
python src/01_download_data.py

# 2. Prepare listing-level modeling table
python src/02_prepare_data.py

# 3. Optional early data quality check after data preparation
python src/03_data_quality_report.py

# 4. Train price prediction models and identify undervalued candidates
python src/04_price_prediction.py

# 5. Cluster undervalued candidates spatially
python src/05_undervalued_clustering.py

# 6. Regenerate final data quality report with model and cluster outputs
python src/03_data_quality_report.py
```

Optional EDA figures:

```bash
python notebooks/eda_pre.py
```

Optional interactive dashboard:

```bash
streamlit run src/app_airbnb_dashboard.py
```

> Note: The dashboard requires Streamlit. If it is not already installed, run:
>
> ```bash
> pip install streamlit
> ```

---

## What Each Script Does

### 1) `src/01_download_data.py`

Downloads the raw source data used in the project.

- Creates the required local data folders if they do not already exist.
- Downloads the raw Airbnb and supporting datasets.
- Saves raw files under:

```text
data/raw/
```

### 2) `src/02_prepare_data.py`

Cleans and prepares the listing-level modeling table.

- Reads raw Airbnb listing, review, calendar, subway, and crime-related files.
- Builds listing-level features for modeling and analysis.
- Merges location, review, calendar, transit, and crime features.
- Saves the prepared modeling table to:

```text
data/processed/nyc_airbnb_hidden_gem_model_table.csv
```

### 3) `src/03_data_quality_report.py`

Builds and saves a markdown data quality report.

- Loads the final undervalued model table if it exists:

```text
data/processed/nyc_airbnb_undervalued_model_table.csv
```

- Otherwise falls back to:

```text
data/processed/nyc_airbnb_hidden_gem_model_table.csv
```

- Saves the report to:

```text
outputs/data_quality_report.md
```

- Reports:
  - Raw files used
  - Final table shape
  - Calendar and review research windows
  - Calendar, review, and complaint coverage
  - Missing-value summary for key fields
  - Legacy diagnostic label distributions
  - Undervalued candidate and cluster distributions
  - Summary statistics for price, rating, transit, crime, and model-output columns
  - Major interpretation caveats

### 4) `src/04_price_prediction.py`

Trains supervised price prediction models and identifies undervalued Airbnb listings.

- Reads:

```text
data/processed/nyc_airbnb_hidden_gem_model_table.csv
```

- Target variable:

```text
log_effective_price
```

- Uses listing, host, review, calendar, transit, crime, and location features.
- Compares multiple regression models:
  - Ridge
  - Random Forest
  - Gradient Boosting
- Selects the best model by RMSE.
- Generates out-of-fold predictions to reduce in-sample prediction leakage.
- Creates model-output columns:
  - `predicted_log_price`
  - `predicted_price`
  - `price_residual_log`
  - `price_gap`
  - `undervaluation_ratio`
  - `undervalued_candidate`
- Saves:

```text
outputs/model_comparison.csv
outputs/undervalued_candidates.csv
data/processed/nyc_airbnb_undervalued_model_table.csv
data/processed/data_dictionary_prepared.csv
```

### 5) `src/05_undervalued_clustering.py`

Clusters undervalued Airbnb candidates into spatial opportunity zones.

- Reads:

```text
data/processed/nyc_airbnb_undervalued_model_table.csv
outputs/undervalued_candidates.csv
```

- Uses latitude and longitude for DBSCAN clustering with haversine distance.
- Assigns each undervalued candidate an `undervalued_cluster` label.
- Treats DBSCAN label `-1` as noise / non-clustered candidates.
- Builds cluster-level summary metrics:
  - Number of listings
  - Median observed effective price
  - Median predicted price
  - Median undervaluation ratio
  - Median rating
  - Most common neighborhoods
  - Median subway distance
  - Median crime intensity
- Saves:

```text
outputs/undervalued_candidates.csv
outputs/undervalued_cluster_summary.csv
data/processed/nyc_airbnb_undervalued_model_table.csv
data/processed/data_dictionary_prepared.csv
```

### Optional) `notebooks/eda_pre.py`

Generates EDA figures for project exploration and presentation support.

- Automatically detects the most relevant processed CSV from:

```text
data/processed/
```

- Creates the following folders if missing:

```text
outputs/
outputs/figures/
```

- Generates figures for:
  - Missing-value patterns
  - Price distributions
  - Feature relationships
  - Calendar availability
  - Review activity
  - Subway accessibility
  - Crime intensity
  - Neighborhood-level price and listing-count patterns
- Saves EDA figures as PNG files under:

```text
outputs/figures/
```

Run with:

```bash
python notebooks/eda_pre.py
```

### Optional) `src/app_airbnb_dashboard.py`

Launches an interactive Streamlit dashboard for user-facing exploration.

- Reads model and clustering outputs:

```text
data/processed/nyc_airbnb_undervalued_model_table.csv
outputs/undervalued_candidates.csv
outputs/undervalued_cluster_summary.csv
outputs/model_comparison.csv
```

- Provides sidebar filters for:
  - Borough
  - Room type
  - Maximum observed price
  - Minimum rating
  - Minimum model-implied discount
  - Minimum review evidence
  - Model-flagged undervalued candidates
  - Spatial cluster ID
- Includes dashboard tabs for:
  - Overview
  - Interactive map
  - Recommendations
  - Key insights
  - Spatial clusters
  - Workflow explanation

Run after completing scripts 01–05:

```bash
streamlit run src/app_airbnb_dashboard.py
```

---

## Recommended Fixes

### Add Streamlit to `requirements.txt`

```text
streamlit
```

### Fix dashboard project root path

In `src/app_airbnb_dashboard.py`, update:

```python
PROJECT_ROOT = APP_DIR
```

to:

```python
PROJECT_ROOT = APP_DIR.parent
```
