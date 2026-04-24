# NYC Airbnb Hidden Gems vs Overpriced Traps

## Research Question
**Identifying Hidden Gems and Overpriced Traps in New York City Airbnb listings from a consumer perspective.**

This project builds a reproducible data acquisition and data preparation pipeline for Airbnb listings in NYC, combining listing metadata, calendar pricing/availability, review activity, subway accessibility, and nearby NYPD complaint activity.

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
Builds a modeling table by:
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
  - Missing-indicator columns
  - Group-comparative quantiles by `neighbourhood_cleansed + room_type`
  - Labels:
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
  - Class distribution
  - Summary stats for core numeric fields
  - Important caveats for interpretation

## Hidden Gem / Overpriced Trap Logic

Comparable groups: `neighbourhood_cleansed + room_type`.

- **Hidden Gem (`hidden_gem_label = 1`)** when listing is low-priced and high-rated/value:
  - `effective_price <= group_price_q25`
  - `review_scores_rating >= group_rating_q75`
  - `review_scores_value >= group_value_median`
  - `review_scores_location >= group_location_median`
  - Sufficient review evidence
  - Better-than-group-median subway proximity (when available)

- **Overpriced Trap (`overpriced_trap_label = 1`)** when listing is high-priced but underperforms:
  - `effective_price >= group_price_q75`
  - AND one of:
    - `review_scores_rating < group_rating_median`
    - `review_scores_value < group_value_median`
    - Worse subway distance than group median
  - Sufficient review evidence

`consumer_value_class`:
- `0`: normal
- `1`: hidden gem
- `2`: overpriced trap

Conflicts are resolved so labels do not overlap.

## Output Files
- `data/interim/calendar_aggregated.csv`
- `data/interim/reviews_aggregated.csv`
- `data/processed/nyc_airbnb_hidden_gem_model_table.csv`
- `data/processed/data_dictionary_prepared.csv`
- `outputs/data_quality_report.md`

## Notes
- If any Inside Airbnb snapshot URL is unavailable (404), update the date constants in scripts to the closest valid snapshot date from the official Get the Data page.
- Calendar prices/availability are scrape-time observations, not finalized transaction records.
- NYPD complaint counts represent reported incidents, not exact true crime rates. They may reflect reporting behavior, police activity, and local population or tourist density.
