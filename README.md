# Spatial-Temporal Discovery of Undervalued Airbnb Listings in NYC

## Project Overview

This project identifies **potentially undervalued Airbnb listings in New York City** by combining listing characteristics, calendar behavior, reviews, subway accessibility, and neighborhood crime context.

The main idea is to estimate each listing's **theoretical market price** using machine learning, then compare the model-predicted price with the observed effective price. Listings with lower observed prices, strong ratings, and enough review evidence are flagged as potential undervalued candidates.

The final project report is included in this repository:

```text
43_project4_final_report.pdf
```

---

## Project Objective

The goal is to build a reproducible spatial-temporal pipeline that answers the following question:

> Can we identify Airbnb listings in NYC that appear underpriced relative to their predicted market value, while still maintaining reasonable quality and review evidence?

A listing is considered a potential opportunity when:

- its observed `effective_price` is meaningfully below its model-predicted price;
- it has sufficient review evidence;
- it maintains strong guest ratings;
- it is located in a meaningful spatial or neighborhood context.

---

## Data Sources

This project combines multiple public data sources:

1. **Inside Airbnb NYC detailed listings**  
   - https://insideairbnb.com/get-the-data/

2. **Inside Airbnb NYC detailed calendar**  
   - URL pattern:  
     `https://data.insideairbnb.com/united-states/ny/new-york-city/YYYY-MM-DD/data/calendar.csv.gz`

3. **Inside Airbnb NYC detailed reviews**  
   - URL pattern:  
     `https://data.insideairbnb.com/united-states/ny/new-york-city/YYYY-MM-DD/data/reviews.csv.gz`

4. **MTA Subway Stations**  
   - Dataset page: https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f  
   - CSV API: https://data.ny.gov/resource/39hk-dx4f.csv?$limit=50000

5. **NYPD Complaint Data**  
   - Historic dataset: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i  
   - Current Year To Date dataset: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243

---

## Repository Structure

```text
43final/
├── data/                         # Raw and processed project data
├── notebooks/                    # EDA and exploratory notebooks/scripts
├── outputs/                      # Model results, figures, and candidate tables
├── src/                          # Main reproducible pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   ├── 03_data_quality_report.py
│   ├── 04_price_prediction.py
│   ├── 05_undervalued_clustering.py
│   └── app_airbnb_dashboard.py
├── 43_project4_final_report.pdf  # Final written report
├── README.md
└── requirements.txt
```

---

## Setup

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows users, activate the environment with:

```bash
.venv\Scripts\activate
```

---

## Run the Reproducible Pipeline

Run the following scripts from the project root.

```bash
# 1. Download raw source data
python src/01_download_data.py

# 2. Prepare listing-level modeling table
python src/02_prepare_data.py

# 3. Generate data quality report
python src/03_data_quality_report.py

# 4. Train price prediction models and identify undervalued candidates
python src/04_price_prediction.py

# 5. Cluster undervalued candidates spatially
python src/05_undervalued_clustering.py

# 6. Regenerate final data quality report with model and cluster outputs
python src/03_data_quality_report.py
```

---

## Optional EDA Figures

To generate EDA figures for exploration and presentation support, run:

```bash
python notebooks/eda_pre.py
```

Figures are saved under:

```text
outputs/figures/
```

---

## Optional Interactive Dashboard

After completing the full pipeline, launch the Streamlit dashboard with:

```bash
streamlit run src/app_airbnb_dashboard.py
```

The dashboard supports interactive exploration of:

- model-flagged undervalued listings;
- borough and room type filters;
- price and rating filters;
- model-implied discount;
- spatial cluster ID;
- interactive map views;
- recommendations and key insights.

If Streamlit is not installed, run:

```bash
pip install streamlit
```

---

## Main Pipeline Components

### `src/01_download_data.py`

Downloads the raw Airbnb, subway, and crime datasets and saves them under:

```text
data/raw/
```

### `src/02_prepare_data.py`

Cleans and merges listing, calendar, review, subway, and crime information into a listing-level modeling table.

Main output:

```text
data/processed/nyc_airbnb_hidden_gem_model_table.csv
```

### `src/03_data_quality_report.py`

Generates a markdown data quality report, including missing-value summaries, coverage checks, and interpretation caveats.

Main output:

```text
outputs/data_quality_report.md
```

### `src/04_price_prediction.py`

Trains supervised price prediction models and identifies undervalued candidates.

Models compared include:

- Ridge Regression
- Random Forest
- Gradient Boosting

Main outputs:

```text
outputs/model_comparison.csv
outputs/undervalued_candidates.csv
data/processed/nyc_airbnb_undervalued_model_table.csv
data/processed/data_dictionary_prepared.csv
```

### `src/05_undervalued_clustering.py`

Clusters undervalued candidates into spatial opportunity zones using DBSCAN with geographic coordinates.

Main outputs:

```text
outputs/undervalued_candidates.csv
outputs/undervalued_cluster_summary.csv
data/processed/nyc_airbnb_undervalued_model_table.csv
data/processed/data_dictionary_prepared.csv
```

### `src/app_airbnb_dashboard.py`

Launches the interactive Streamlit dashboard using the final model and clustering outputs.

---

## Final Outputs

Key project outputs include:

```text
43_project4_final_report.pdf
outputs/model_comparison.csv
outputs/undervalued_candidates.csv
outputs/undervalued_cluster_summary.csv
outputs/data_quality_report.md
outputs/figures/
```

---

## Notes

- Some raw data files may be large. If large files are not included in the repository, they can be regenerated by running `src/01_download_data.py`.
- The project uses public data sources and is designed to be reproducible from the project root.
- The final report provides the complete project motivation, methodology, modeling results, and interpretation.

