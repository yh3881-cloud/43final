# NYC Airbnb Hidden Gems vs Overpriced Traps

## Research Question
**Identifying Hidden Gems and Overpriced Traps in New York City Airbnb listings from a consumer perspective.**

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
## Data Preprocessing

In `src/02_prepare_data.py`, the **data preprocessing** stage includes the following operations.

### 1. Robust file detection and loading
- Automatically detects raw files in `data/raw` using keyword matching:
  - required: listings, calendar, reviews, subway
  - optional: crime/complaint/NYPD
- Raises clear errors for missing required files and skips optional crime data if unavailable.

### 2. Listings cleaning (base table)
- Converts `id` and `host_id` to standardized string keys.
- Normalizes listing keys (trim spaces, remove trailing `.0`).
- Cleans `price` from currency strings to numeric values.
- Converts percentage fields (e.g., `host_response_rate`, `host_acceptance_rate`) to decimals.
- Converts boolean-like text (`t/f`, `true/false`) into binary indicators.
- Parses amenities text and creates `amenity_count`.
- Converts latitude/longitude to numeric.
- Keeps target modeling columns when available; creates missing columns as `NaN` if absent.
- Removes invalid listing rows:
  - missing ID
  - missing/non-positive price
  - missing coordinates

### 3. Calendar preprocessing before aggregation
- Standardizes `listing_id` keys to align with listings.
- Parses date and filters to the research window (`2025-04-01` to `2026-03-31`).
- Builds `calendar_price` with row-level fallback:
  - use `adjusted_price` if present, otherwise `price`
- Converts availability to numeric `is_available`.
- Creates weekend indicator (Friday and Saturday nights).

### 4. Reviews preprocessing before aggregation
- Standardizes `listing_id` keys.
- Parses review dates and filters to the same research window.
- Selects reviewer identity column flexibly (`reviewer_id` preferred, otherwise `reviewer_name`).

### 5. Subway data cleaning
- Detects coordinate columns flexibly (`gtfs_latitude/gtfs_longitude`, `latitude/longitude`, `lat/lon/long`).
- Parses WKT/geospatial point text when needed.
- Drops stations with invalid/missing coordinates.

### 6. Optional crime data cleaning
- Detects complaint date column flexibly (e.g., `cmplnt_fr_dt`, `complaint_date`, `date`).
- Filters to the research window.
- Converts and validates coordinates.
- Restricts records to approximate NYC bounds.
- Standardizes offense and law-category text.

### 7. Merge-safe postprocessing
- Uses listings as base table.
- Left joins aggregated calendar and reviews, then subway and optional crime features.
- Drops redundant merge keys.
- Creates:
  - `effective_price` (calendar median price fallback to listing price)
  - `missing_calendar_data` indicator
- Applies controlled missing handling:
  - fills review/crime count features with 0
  - fills key categorical fields with `"Unknown"`
  - creates missing-indicator columns for important numeric variables

---

## Feature Engineering

The **feature engineering** stage transforms cleaned data into listing-level predictive signals and target labels.

### 1. Calendar-derived features (listing level)
From daily calendar records:
- `calendar_days`
- `calendar_avg_price`
- `calendar_median_price`
- `calendar_min_price`
- `calendar_max_price`
- `calendar_price_std`
- `calendar_price_volatility`
- `calendar_available_days`
- `calendar_available_rate`
- `calendar_unavailable_days`
- `calendar_weekend_avg_price`
- `calendar_weekday_avg_price`
- `weekend_price_premium`
- `calendar_minimum_nights_median`

### 2. Review-derived features (listing level)
From review-level records:
- `reviews_in_window`
- `unique_reviewers_in_window`
- `first_review_in_window`
- `last_review_in_window`
- `reviews_last_90d`
- `reviews_last_180d`
- `reviews_last_365d`
- `days_since_last_review_in_window`

### 3. Subway accessibility features (geospatial)
Using BallTree + haversine:
- `distance_to_nearest_subway_km`
- `subway_stations_within_500m`
- `subway_stations_within_1000m`

### 4. Optional crime proximity features (geospatial)
If crime data exists:
- `crime_count_500m`
- `crime_count_1000m`
- `violent_crime_count_1000m`
- `property_crime_count_1000m`
- `felony_count_1000m`
- `misdemeanor_count_1000m`
- `violation_count_1000m`
- `crime_intensity_log_1000m = log1p(crime_count_1000m)`

### 5. Missingness features
Creates binary indicators such as:
- `missing_review_scores_rating`
- `missing_review_scores_cleanliness`
- `missing_review_scores_location`
- `missing_review_scores_value`
- `missing_host_response_rate`
- `missing_host_acceptance_rate`
- `missing_bedrooms`
- `missing_beds`
- `missing_bathrooms`
- `missing_calendar_median_price`
- `missing_calendar_available_rate`

### 6. Group-relative benchmark features
Builds comparison groups by `neighbourhood_cleansed + room_type`, then computes:
- `group_price_q25`
- `group_price_q75`
- `group_rating_q75`
- `group_rating_median`
- `group_value_median`
- `group_location_median`
- `group_subway_distance_median`
- `group_crime_intensity_median` (if crime exists)

### 7. Label engineering for consumer-value classes
- Defines `has_enough_reviews`:
  - `number_of_reviews >= 5` OR `reviews_in_window >= 2`
- Creates:
  - `hidden_gem_label`
  - `overpriced_trap_label`
- Prevents overlap by prioritizing hidden-gem assignments when both are triggered.
- Creates final multiclass target:
  - `consumer_value_class` (`0=normal`, `1=hidden gem`, `2=overpriced trap`)



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

