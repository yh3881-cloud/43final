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


## Notes
- If any Inside Airbnb snapshot URL is unavailable (404), update the date constants in scripts to the closest valid snapshot date from the official Get the Data page.
- Calendar prices/availability are scrape-time observations, not finalized transaction records.
- NYPD complaint counts represent reported incidents, not exact true crime rates. They may reflect reporting behavior, police activity, and local population or tourist density.
