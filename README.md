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

## Snapshot Files and Time Window
- Listings snapshot date: `2025-04-01`
- Calendar snapshot date: `2025-04-01`
- Reviews snapshot date: `2026-04-01`
- Research window: `2025-04-01` to `2026-03-31`

### Why these dates?
- `listings` is treated as a market snapshot near the start of the research period.
- `calendar` is taken from the same snapshot and filtered to the next 365 days (`2025-04-01` to `2026-03-31`).
- `reviews` is pulled from a post-window snapshot (`2026-04-01`) so reviews inside the target window can be recovered.
- NYPD complaint records are pulled with an API-side date filter to the exact study window, avoiding unnecessary years.

## Project Structure
```text
airbnb_nyc_hidden_gem/
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/
    01_download_data.py
    02_prepare_data.py
    03_data_quality_report.py
  outputs/
  README.md
  requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/01_download_data.py
python src/02_prepare_data.py
python src/03_data_quality_report.py
python src/04_price_prediction.py
python src/05_undervalued_clustering.py
```

## What Each Script Does

### 1) `src/01_download_data.py`
- Creates `data/raw/` if missing.
- Downloads:
  - `data/raw/listings_2025-04-01.csv.gz`
  - `data/raw/calendar_2025-04-01.csv.gz`
  - `data/raw/reviews_2026-04-01.csv.gz`
  - `data/raw/mta_subway_stations.csv`
  - `data/raw/nypd_complaints_2025-04-01_2026-03-31.csv`
- Uses streaming download (`requests` + `tqdm` progress bar).
- Performs file-size and non-empty checks.
- Handles 404 with clear guidance to switch to the nearest available snapshot date from Inside Airbnb Get the Data page.
- Pulls NYPD complaint records via Socrata API with `cmplnt_fr_dt` filter between `2025-04-01` and `2026-03-31`.

### 2) `src/02_prepare_data.py`
Builds the integrated listing-level feature table by:
- Cleaning listings:
  - Currency, percentages, `t/f` booleans, amenities count, invalid row filtering.
  - Keeps ML-focused features and adds outlier price flags.
- Cleaning calendar:
  - Filters to research window.
  - Creates listing-level price/availability and weekend premium features.
- Cleaning reviews:
  - Filters to research window.
  - Aggregates activity/intensity and recency features.
- Cleaning subway stations:
  - Detects coordinate columns (including WKT parsing where needed).
  - Computes nearest subway distance and nearby station counts with `cKDTree` + haversine.
- Cleaning NYPD complaints:
  - Keeps only: `cmplnt_num`, `cmplnt_fr_dt`, `cmplnt_fr_tm`, `ofns_desc`, `law_cat_cd`, `boro_nm`, `latitude`, `longitude`
  - Parses complaint date, standardizes offense and law category fields.
  - Drops missing/invalid coordinates and filters to approximate NYC coordinate bounds.
  - Creates broad offense groups: `violent_crime`, `property_crime`, `other_crime`.
- Crime proximity features per listing (haversine distance with optimized candidate search):
  - `crime_count_500m`, `crime_count_1000m`
  - `felony_count_1000m`, `misdemeanor_count_1000m`, `violation_count_1000m`
  - `violent_crime_count_1000m`, `property_crime_count_1000m`, `other_crime_count_1000m`
  - `crime_intensity_log_1000m = log1p(crime_count_1000m)`
- Merging all features and creating:
  - `effective_price` (calendar median price fallback to listing price)
  - `log_effective_price = log1p(effective_price)`
  - Missing-indicator columns
  - Legacy diagnostic labels (kept for reference):
    - `hidden_gem_label`
    - `overpriced_trap_label`
    - `consumer_value_class` (0 normal, 1 hidden gem, 2 overpriced trap)

### 3) `src/03_data_quality_report.py`
- Loads the processed table.
- Prints and saves `outputs/data_quality_report.md`.
- Reports:
  - Raw files used
  - Final shape
  - Date windows
  - Missing summary (key fields)
  - Calendar/review coverage
  - Legacy class distribution and undervalued columns when available
  - Summary stats for core numeric fields
  - Important caveats for interpretation

### 4) `src/04_price_prediction.py`
- Loads the prepared listing-level table.
- Trains three regressors on `log_effective_price`:
  - Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluates models using MAE, RMSE, R2.
- Selects the best model by RMSE and generates:
  - `predicted_log_price`
  - `predicted_price`
  - `price_residual_log = predicted_log_price - log_effective_price`
  - `price_gap = predicted_price - effective_price`
  - `undervaluation_ratio = predicted_price / effective_price`
- Creates:
  - `undervalued_candidate = 1` when:
    - residual is above configurable percentile (default top 10%)
    - `review_scores_rating >= 4.8`
    - `number_of_reviews >= 5` OR `reviews_in_window >= 2`
    - `effective_price > 0`
- Saves:
  - `data/processed/nyc_airbnb_undervalued_model_table.csv`
  - `outputs/model_comparison.csv`
  - `outputs/undervalued_candidates.csv`

### 5) `src/05_undervalued_clustering.py`
- Loads undervalued candidates.
- Applies DBSCAN on latitude/longitude.
- Creates `undervalued_cluster`.
- Saves:
  - `outputs/undervalued_cluster_summary.csv` with:
    - `cluster_id`
    - `number_of_listings`
    - `median_effective_price`
    - `median_predicted_price`
    - `median_undervaluation_ratio`
    - `median_rating`
    - `most_common_neighbourhoods`
    - `median_subway_distance`
    - `median_crime_intensity`

## Output Files
- `data/interim/calendar_aggregated.csv`
- `data/interim/reviews_aggregated.csv`
- `data/processed/nyc_airbnb_hidden_gem_model_table.csv` (base integrated table)
- `data/processed/nyc_airbnb_undervalued_model_table.csv` (prediction-enriched table)
- `data/processed/data_dictionary_prepared.csv`
- `outputs/data_quality_report.md`
- `outputs/model_comparison.csv`
- `outputs/undervalued_candidates.csv`
- `outputs/undervalued_cluster_summary.csv`

## Notes
- If any Inside Airbnb snapshot URL is unavailable (404), update the date constants in scripts to the closest valid snapshot date from the official Get the Data page.
- Calendar prices/availability are scrape-time observations, not finalized transaction records.
- NYPD complaint counts represent reported incidents, not exact true crime rates. They may reflect reporting behavior, police activity, and local population or tourist density.
