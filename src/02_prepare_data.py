from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


RESEARCH_START = "2025-04-01"
RESEARCH_END = "2026-03-31"
EARTH_RADIUS_KM = 6371.0088

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FINAL_TABLE_PATH = PROCESSED_DIR / "nyc_airbnb_hidden_gem_model_table.csv"
DATA_DICT_PATH = PROCESSED_DIR / "data_dictionary_prepared.csv"
CALENDAR_AGG_PATH = INTERIM_DIR / "calendar_aggregated.csv"
REVIEWS_AGG_PATH = INTERIM_DIR / "reviews_aggregated.csv"
SUMMARY_REPORT_PATH = OUTPUTS_DIR / "data_integration_summary.md"

NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.65

LISTING_COLUMNS_TARGET = [
    "id", "listing_url", "name", "description", "host_id", "host_since",
    "host_is_superhost", "host_response_rate", "host_acceptance_rate",
    "host_listings_count", "host_total_listings_count", "host_identity_verified",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms", "bathrooms_text",
    "bedrooms", "beds", "amenities", "amenity_count", "price", "minimum_nights",
    "maximum_nights", "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "estimated_occupancy_l365d", "estimated_revenue_l365d",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value", "instant_bookable", "calculated_host_listings_count",
    "reviews_per_month",
]

VIOLENT_CRIME_SET = {
    "ROBBERY",
    "FELONY ASSAULT",
    "RAPE",
    "MURDER & NON-NEGL. MANSLAUGHTER",
}
PROPERTY_CRIME_SET = {
    "BURGLARY",
    "GRAND LARCENY",
    "PETIT LARCENY",
    "GRAND LARCENY OF MOTOR VEHICLE",
}


def ensure_dirs() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_currency_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def clean_percent_to_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan}).str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce") / 100.0


def tf_to_binary(series: pd.Series) -> pd.Series:
    mapping = {"t": 1, "f": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)


def parse_amenity_count(raw_value: object) -> int:
    if pd.isna(raw_value):
        return 0
    text = str(raw_value).strip().strip("{}[]")
    if not text:
        return 0
    parts = [x.strip().strip('"').strip("'") for x in text.split(",")]
    return int(sum(1 for x in parts if x))


def parse_point_wkt(value: object) -> Tuple[float, float]:
    if pd.isna(value):
        return (np.nan, np.nan)
    match = re.search(r"POINT\s*\(\s*([\-0-9.]+)\s+([\-0-9.]+)\s*\)", str(value))
    if not match:
        return (np.nan, np.nan)
    return float(match.group(2)), float(match.group(1))


def normalize_listing_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )


def detect_file(candidates: List[Path], keywords: List[str], exclude_keywords: Optional[List[str]] = None) -> Optional[Path]:
    best_path = None
    best_score = -1
    exclude_keywords = exclude_keywords or []
    for path in candidates:
        name = path.name.lower()
        if any(k in name for k in exclude_keywords):
            continue
        score = sum(2 for k in keywords if k in name)
        if path.suffix in {".csv", ".gz"}:
            score += 1
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_score > 0 else None


def detect_raw_files(raw_dir: Path) -> Dict[str, Optional[Path]]:
    candidates = [p for p in raw_dir.iterdir() if p.is_file()]
    file_map = {
        "listings": detect_file(candidates, ["listing", "listings"]),
        "calendar": detect_file(candidates, ["calendar"]),
        "reviews": detect_file(candidates, ["review", "reviews"], exclude_keywords=["dictionary"]),
        "subway": detect_file(candidates, ["subway", "station", "mta"]),
        "crime": detect_file(candidates, ["crime", "complaint", "nypd"]),
    }
    return file_map


def require_file(file_path: Optional[Path], name: str) -> Path:
    if file_path is None:
        raise FileNotFoundError(f"Required raw file for '{name}' not found in {RAW_DIR}.")
    return file_path


def read_csv_auto(path: Path) -> pd.DataFrame:
    print(f"[INFO] Using file: {path}")
    return pd.read_csv(path, low_memory=False)


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def prepare_listings(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Listings file must contain `id` column.")
    out = df.copy()
    out["id"] = normalize_listing_key(out["id"])
    if "host_id" in out.columns:
        out["host_id"] = out["host_id"].astype(str)

    if "price" not in out.columns:
        raise ValueError("Listings file must contain `price` column.")
    out["price"] = clean_currency_to_float(out["price"])

    for col in ["host_response_rate", "host_acceptance_rate"]:
        out[col] = clean_percent_to_float(out[col]) if col in out.columns else np.nan
    for col in ["host_is_superhost", "instant_bookable", "host_identity_verified"]:
        out[col] = tf_to_binary(out[col]) if col in out.columns else np.nan

    out["amenities"] = out["amenities"] if "amenities" in out.columns else np.nan
    out["amenity_count"] = out["amenities"].apply(parse_amenity_count)
    out["latitude"] = pd.to_numeric(out.get("latitude"), errors="coerce")
    out["longitude"] = pd.to_numeric(out.get("longitude"), errors="coerce")

    for col in LISTING_COLUMNS_TARGET:
        if col not in out.columns:
            out[col] = np.nan
    out = out[LISTING_COLUMNS_TARGET].copy()

    before = len(out)
    out = out.dropna(subset=["id", "price", "latitude", "longitude"])
    out = out[out["price"] > 0].copy()
    print(f"[INFO] Listings retained after cleaning: {len(out)}/{before}")
    return out


def prepare_calendar(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    if "listing_id" not in df.columns or "date" not in df.columns:
        raise ValueError("Calendar file must contain `listing_id` and `date` columns.")
    out = df.copy()
    out["listing_id"] = normalize_listing_key(out["listing_id"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].between(start, end, inclusive="both")].copy()
    rows_after_filter = len(out)

    # Row-level fallback: use adjusted_price when present, otherwise price.
    price_raw = pd.Series(np.nan, index=out.index, dtype="object")
    if "adjusted_price" in out.columns:
        price_raw = out["adjusted_price"]
    if "price" in out.columns:
        price_raw = price_raw.where(price_raw.notna(), out["price"])
    out["calendar_price"] = clean_currency_to_float(price_raw)
    out["is_available"] = tf_to_binary(out["available"]) if "available" in out.columns else np.nan
    out["is_weekend"] = out["date"].dt.dayofweek.isin([4, 5]).astype(int)

    grouped = out.groupby("listing_id", dropna=False)
    agg = grouped.agg(
        calendar_days=("date", "count"),
        calendar_avg_price=("calendar_price", "mean"),
        calendar_median_price=("calendar_price", "median"),
        calendar_min_price=("calendar_price", "min"),
        calendar_max_price=("calendar_price", "max"),
        calendar_price_std=("calendar_price", "std"),
        calendar_available_days=("is_available", "sum"),
        calendar_minimum_nights_median=("minimum_nights", "median"),
    ).reset_index()

    agg["calendar_price_volatility"] = np.where(
        agg["calendar_avg_price"] > 0,
        agg["calendar_price_std"] / agg["calendar_avg_price"],
        np.nan,
    )
    agg["calendar_unavailable_days"] = agg["calendar_days"] - agg["calendar_available_days"]
    agg["calendar_available_rate"] = np.where(
        agg["calendar_days"] > 0, agg["calendar_available_days"] / agg["calendar_days"], np.nan
    )
    weekend_avg = out[out["is_weekend"] == 1].groupby("listing_id")["calendar_price"].mean().rename("calendar_weekend_avg_price")
    weekday_avg = out[out["is_weekend"] == 0].groupby("listing_id")["calendar_price"].mean().rename("calendar_weekday_avg_price")
    agg = agg.merge(weekend_avg, on="listing_id", how="left").merge(weekday_avg, on="listing_id", how="left")
    agg["weekend_price_premium"] = np.where(
        agg["calendar_weekday_avg_price"] > 0,
        (agg["calendar_weekend_avg_price"] - agg["calendar_weekday_avg_price"]) / agg["calendar_weekday_avg_price"],
        np.nan,
    )
    return agg, rows_after_filter


def prepare_reviews(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    if "listing_id" not in df.columns or "date" not in df.columns:
        raise ValueError("Reviews file must contain `listing_id` and `date` columns.")
    out = df.copy()
    out["listing_id"] = normalize_listing_key(out["listing_id"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].between(start, end, inclusive="both")].copy()
    rows_after_filter = len(out)

    reviewer_col = find_first_existing_column(out, ["reviewer_id", "reviewer_name"])
    grouped = out.groupby("listing_id", dropna=False)
    agg = grouped.agg(
        reviews_in_window=("date", "count"),
        first_review_in_window=("date", "min"),
        last_review_in_window=("date", "max"),
    ).reset_index()
    agg["unique_reviewers_in_window"] = grouped[reviewer_col].nunique().values if reviewer_col else 0

    d90 = (end - pd.Timedelta(days=90))
    d180 = (end - pd.Timedelta(days=180))
    d365 = (end - pd.Timedelta(days=365))
    agg["reviews_last_90d"] = grouped["date"].apply(lambda s: int((s >= d90).sum())).values
    agg["reviews_last_180d"] = grouped["date"].apply(lambda s: int((s >= d180).sum())).values
    agg["reviews_last_365d"] = grouped["date"].apply(lambda s: int((s >= d365).sum())).values
    agg["days_since_last_review_in_window"] = (end - pd.to_datetime(agg["last_review_in_window"], errors="coerce")).dt.days
    return agg, rows_after_filter


def detect_coordinates(df: pd.DataFrame, lat_candidates: List[str], lon_candidates: List[str]) -> Tuple[pd.Series, pd.Series]:
    lat_col = find_first_existing_column(df, lat_candidates)
    lon_col = find_first_existing_column(df, lon_candidates)
    if lat_col and lon_col:
        return pd.to_numeric(df[lat_col], errors="coerce"), pd.to_numeric(df[lon_col], errors="coerce")

    geo_col = None
    for c in df.columns:
        if "geo" in c.lower() or "point" in c.lower() or "location" in c.lower():
            geo_col = c
            break
    if geo_col is None:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    parsed = df[geo_col].apply(parse_point_wkt)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["lat", "lon"], index=df.index)
    return parsed_df["lat"], parsed_df["lon"]


def build_balltree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    coords_rad = np.radians(np.column_stack([lat, lon]))
    return BallTree(coords_rad, metric="haversine")


def query_count_within_radius(tree: BallTree, query_lat: np.ndarray, query_lon: np.ndarray, radius_km: float) -> np.ndarray:
    query_rad = np.radians(np.column_stack([query_lat, query_lon]))
    radius_rad = radius_km / EARTH_RADIUS_KM
    idx = tree.query_radius(query_rad, r=radius_rad, count_only=False)
    return np.array([len(x) for x in idx], dtype=int)


def query_nearest_distance_km(tree: BallTree, query_lat: np.ndarray, query_lon: np.ndarray) -> np.ndarray:
    query_rad = np.radians(np.column_stack([query_lat, query_lon]))
    dist_rad, _ = tree.query(query_rad, k=1)
    return dist_rad.reshape(-1) * EARTH_RADIUS_KM


def build_subway_features(listings_df: pd.DataFrame, subway_df: pd.DataFrame) -> pd.DataFrame:
    out = listings_df[["id", "latitude", "longitude"]].copy()
    if subway_df.empty:
        out["distance_to_nearest_subway_km"] = np.nan
        out["subway_stations_within_500m"] = 0
        out["subway_stations_within_1000m"] = 0
        return out
    tree = build_balltree(subway_df["subway_latitude"].to_numpy(), subway_df["subway_longitude"].to_numpy())
    qlat = out["latitude"].to_numpy()
    qlon = out["longitude"].to_numpy()
    out["distance_to_nearest_subway_km"] = query_nearest_distance_km(tree, qlat, qlon)
    out["subway_stations_within_500m"] = query_count_within_radius(tree, qlat, qlon, 0.5)
    out["subway_stations_within_1000m"] = query_count_within_radius(tree, qlat, qlon, 1.0)
    return out


def prepare_subway(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lat, lon = detect_coordinates(out, ["gtfs_latitude", "latitude", "lat"], ["gtfs_longitude", "longitude", "lon", "long"])
    out["subway_latitude"] = lat
    out["subway_longitude"] = lon
    before = len(out)
    out = out.dropna(subset=["subway_latitude", "subway_longitude"]).copy()
    print(f"[INFO] Subway rows retained: {len(out)}/{before}")
    return out


def prepare_crime(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    date_col = find_first_existing_column(out, ["cmplnt_fr_dt", "complaint_date", "occurrence_date", "date"])
    if date_col is None:
        raise ValueError("Crime file found but no date column detected.")
    lat, lon = detect_coordinates(out, ["latitude", "lat"], ["longitude", "lon", "long"])
    out["crime_latitude"] = lat
    out["crime_longitude"] = lon
    out["crime_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out["crime_date"].between(start, end, inclusive="both")].copy()
    out = out.dropna(subset=["crime_latitude", "crime_longitude"]).copy()
    out = out[
        (out["crime_latitude"] >= NYC_LAT_MIN) & (out["crime_latitude"] <= NYC_LAT_MAX) &
        (out["crime_longitude"] >= NYC_LON_MIN) & (out["crime_longitude"] <= NYC_LON_MAX)
    ].copy()
    rows_after_filter = len(out)

    ofns_col = find_first_existing_column(out, ["ofns_desc", "offense_description", "offense", "ofns"])
    law_col = find_first_existing_column(out, ["law_cat_cd", "law_category", "law_cat", "category"])
    if ofns_col:
        out["ofns_desc_std"] = out[ofns_col].astype(str).str.upper().str.strip()
    else:
        out["ofns_desc_std"] = np.nan
    if law_col:
        out["law_cat_cd_std"] = out[law_col].astype(str).str.upper().str.strip()
    else:
        out["law_cat_cd_std"] = np.nan

    out["violent_crime_flag"] = out["ofns_desc_std"].isin(VIOLENT_CRIME_SET).astype(int)
    out["property_crime_flag"] = out["ofns_desc_std"].isin(PROPERTY_CRIME_SET).astype(int)
    out["felony_flag"] = (out["law_cat_cd_std"] == "FELONY").astype(int)
    out["misdemeanor_flag"] = (out["law_cat_cd_std"] == "MISDEMEANOR").astype(int)
    out["violation_flag"] = (out["law_cat_cd_std"] == "VIOLATION").astype(int)
    return out, rows_after_filter


def counts_with_flag(listings_df: pd.DataFrame, points_df: pd.DataFrame, radius_km: float, flag_col: Optional[str] = None) -> np.ndarray:
    if points_df.empty:
        return np.zeros(len(listings_df), dtype=int)
    subset = points_df if flag_col is None else points_df[points_df[flag_col] == 1]
    if subset.empty:
        return np.zeros(len(listings_df), dtype=int)
    tree = build_balltree(subset["crime_latitude"].to_numpy(), subset["crime_longitude"].to_numpy())
    return query_count_within_radius(tree, listings_df["latitude"].to_numpy(), listings_df["longitude"].to_numpy(), radius_km)


def build_crime_features(listings_df: pd.DataFrame, crime_df: pd.DataFrame) -> pd.DataFrame:
    out = listings_df[["id", "latitude", "longitude"]].copy()
    out["crime_count_500m"] = counts_with_flag(out, crime_df, 0.5, None)
    out["crime_count_1000m"] = counts_with_flag(out, crime_df, 1.0, None)
    out["violent_crime_count_1000m"] = counts_with_flag(out, crime_df, 1.0, "violent_crime_flag")
    out["property_crime_count_1000m"] = counts_with_flag(out, crime_df, 1.0, "property_crime_flag")
    out["felony_count_1000m"] = counts_with_flag(out, crime_df, 1.0, "felony_flag")
    out["misdemeanor_count_1000m"] = counts_with_flag(out, crime_df, 1.0, "misdemeanor_flag")
    out["violation_count_1000m"] = counts_with_flag(out, crime_df, 1.0, "violation_flag")
    out["crime_intensity_log_1000m"] = np.log1p(out["crime_count_1000m"])
    return out


def add_missing_indicators(df: pd.DataFrame, columns: Iterable[str], prefix: str = "missing_") -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[f"{prefix}{col}"] = df[col].isna().astype(int)
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comparison_group"] = out[["neighbourhood_cleansed", "room_type"]].astype(str).agg(" | ".join, axis=1)
    group_cols = ["neighbourhood_cleansed", "room_type"]
    grp = out.groupby(group_cols, dropna=False)

    out["group_price_q25"] = grp["effective_price"].transform(lambda s: s.quantile(0.25))
    out["group_price_q75"] = grp["effective_price"].transform(lambda s: s.quantile(0.75))
    out["group_rating_q75"] = grp["review_scores_rating"].transform(lambda s: s.quantile(0.75))
    out["group_rating_median"] = grp["review_scores_rating"].transform("median")
    out["group_value_median"] = grp["review_scores_value"].transform("median")
    out["group_location_median"] = grp["review_scores_location"].transform("median")
    out["group_subway_distance_median"] = grp["distance_to_nearest_subway_km"].transform("median")
    if "crime_intensity_log_1000m" in out.columns:
        out["group_crime_intensity_median"] = grp["crime_intensity_log_1000m"].transform("median")

    out["has_enough_reviews"] = (out["number_of_reviews"].fillna(0) >= 5) | (out["reviews_in_window"].fillna(0) >= 2)
    hidden = (
        (out["effective_price"] <= out["group_price_q25"]) &
        (out["review_scores_rating"] >= out["group_rating_q75"]) &
        (out["review_scores_value"] >= out["group_value_median"]) &
        (out["review_scores_location"] >= out["group_location_median"]) &
        (out["distance_to_nearest_subway_km"] <= out["group_subway_distance_median"]) &
        out["has_enough_reviews"]
    )
    if "crime_intensity_log_1000m" in out.columns:
        hidden = hidden & (out["crime_intensity_log_1000m"] <= out["group_crime_intensity_median"])

    trap = (
        (out["effective_price"] >= out["group_price_q75"]) &
        (
            (out["review_scores_rating"] < out["group_rating_median"]) |
            (out["review_scores_value"] < out["group_value_median"]) |
            (out["distance_to_nearest_subway_km"] > out["group_subway_distance_median"])
        ) &
        out["has_enough_reviews"]
    )
    if "crime_intensity_log_1000m" in out.columns:
        trap = trap | (
            (out["effective_price"] >= out["group_price_q75"]) &
            (out["crime_intensity_log_1000m"] > out["group_crime_intensity_median"]) &
            out["has_enough_reviews"]
        )

    out["hidden_gem_label"] = hidden.astype(int)
    out["overpriced_trap_label"] = trap.astype(int)
    out.loc[out["hidden_gem_label"] == 1, "overpriced_trap_label"] = 0
    out["consumer_value_class"] = 0
    out.loc[out["hidden_gem_label"] == 1, "consumer_value_class"] = 1
    out.loc[out["overpriced_trap_label"] == 1, "consumer_value_class"] = 2
    return out


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round((missing_count / n * 100.0), 4) if n else np.nan,
                "sample_values": " | ".join(df[col].dropna().astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def write_summary_report(metrics: Dict[str, object]) -> None:
    report = f"""# Data Integration Summary

## Files Used
- Listings: `{metrics['listings_file']}`
- Calendar: `{metrics['calendar_file']}`
- Reviews: `{metrics['reviews_file']}`
- Subway: `{metrics['subway_file']}`
- Crime: `{metrics['crime_file']}`

## Pipeline Summary
- Listings loaded: **{metrics['listings_loaded']}**
- Listings after cleaning: **{metrics['listings_cleaned']}**
- Calendar rows in window: **{metrics['calendar_rows_window']}**
- Review rows in window: **{metrics['reviews_rows_window']}**
- Crime rows in window: **{metrics['crime_rows_window']}**
- Final table shape: **{metrics['final_shape']}**
- Listings with calendar data: **{metrics['listings_with_calendar']} ({metrics['listings_with_calendar_pct']:.2f}%)**
- Listings with reviews in window: **{metrics['listings_with_reviews']} ({metrics['listings_with_reviews_pct']:.2f}%)**

## Consumer Value Class Distribution
{metrics['class_distribution_md']}

## Top 10 Missing Columns
{metrics['top_missing_md']}
"""
    SUMMARY_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    print("[INFO] Starting robust listing-level integration pipeline...")
    ensure_dirs()
    start = pd.to_datetime(RESEARCH_START)
    end = pd.to_datetime(RESEARCH_END)

    file_map = detect_raw_files(RAW_DIR)
    listings_path = require_file(file_map["listings"], "listings")
    calendar_path = require_file(file_map["calendar"], "calendar")
    reviews_path = require_file(file_map["reviews"], "reviews")
    subway_path = require_file(file_map["subway"], "subway")
    crime_path = file_map["crime"]

    listings_raw = read_csv_auto(listings_path)
    calendar_raw = read_csv_auto(calendar_path)
    reviews_raw = read_csv_auto(reviews_path)
    subway_raw = read_csv_auto(subway_path)

    listings_loaded = len(listings_raw)
    listings = prepare_listings(listings_raw)
    calendar_agg, calendar_rows_window = prepare_calendar(calendar_raw, start, end)
    reviews_agg, reviews_rows_window = prepare_reviews(reviews_raw, start, end)
    subway_clean = prepare_subway(subway_raw)
    subway_features = build_subway_features(listings, subway_clean)

    CALENDAR_AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    calendar_agg.to_csv(CALENDAR_AGG_PATH, index=False)
    reviews_agg.to_csv(REVIEWS_AGG_PATH, index=False)
    print(f"[INFO] Saved calendar aggregation: {CALENDAR_AGG_PATH}")
    print(f"[INFO] Saved reviews aggregation: {REVIEWS_AGG_PATH}")

    merged = listings.merge(calendar_agg, left_on="id", right_on="listing_id", how="left")
    merged = merged.merge(reviews_agg, left_on="id", right_on="listing_id", how="left", suffixes=("", "_reviews"))
    merged = merged.merge(subway_features.drop(columns=["latitude", "longitude"]), on="id", how="left")

    crime_rows_window = 0
    if crime_path is not None:
        try:
            crime_raw = read_csv_auto(crime_path)
            crime_clean, crime_rows_window = prepare_crime(crime_raw, start, end)
            crime_features = build_crime_features(listings, crime_clean)
            merged = merged.merge(crime_features.drop(columns=["latitude", "longitude"]), on="id", how="left")
            print(f"[INFO] Crime features merged from: {crime_path}")
        except Exception as exc:
            print(f"[WARNING] Crime file found but skipped due to issue: {exc}")
    else:
        print("[WARNING] Optional crime/complaint file not found. Skipping crime features.")

    for key_col in ["listing_id", "listing_id_reviews"]:
        if key_col in merged.columns:
            merged = merged.drop(columns=[key_col])

    merged["effective_price"] = merged["calendar_median_price"].where(merged["calendar_median_price"].notna(), merged["price"])
    merged["log_effective_price"] = np.log1p(merged["effective_price"])
    merged["missing_calendar_data"] = merged["calendar_median_price"].isna().astype(int)

    review_count_cols = ["reviews_in_window", "unique_reviewers_in_window", "reviews_last_90d", "reviews_last_180d", "reviews_last_365d"]
    for col in review_count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    crime_count_cols = [
        "crime_count_500m", "crime_count_1000m", "violent_crime_count_1000m",
        "property_crime_count_1000m", "felony_count_1000m", "misdemeanor_count_1000m", "violation_count_1000m",
    ]
    for col in crime_count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    if "crime_count_1000m" in merged.columns and "crime_intensity_log_1000m" not in merged.columns:
        merged["crime_intensity_log_1000m"] = np.log1p(merged["crime_count_1000m"])

    for c in ["neighbourhood_cleansed", "neighbourhood_group_cleansed", "property_type", "room_type"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna("Unknown").replace("", "Unknown")

    merged = add_missing_indicators(merged, ["review_scores_rating", "review_scores_cleanliness", "review_scores_location", "review_scores_value"])
    merged = add_missing_indicators(merged, ["host_response_rate", "host_acceptance_rate"])
    merged = add_missing_indicators(merged, ["bedrooms", "beds", "bathrooms", "calendar_median_price", "calendar_available_rate"])

    labeled = build_labels(merged)
    labeled.to_csv(FINAL_TABLE_PATH, index=False)
    build_data_dictionary(labeled).to_csv(DATA_DICT_PATH, index=False)

    total = len(labeled)
    listings_with_calendar = int(labeled["calendar_median_price"].notna().sum()) if "calendar_median_price" in labeled else 0
    listings_with_reviews = int((labeled.get("reviews_in_window", pd.Series(0, index=labeled.index)) > 0).sum())
    class_distribution_md = labeled["consumer_value_class"].value_counts(dropna=False).sort_index().to_markdown()
    top_missing = (
        labeled.isna().mean().mul(100).sort_values(ascending=False).head(10).rename("missing_pct").round(2).to_frame()
    )

    metrics = {
        "listings_file": listings_path.name,
        "calendar_file": calendar_path.name,
        "reviews_file": reviews_path.name,
        "subway_file": subway_path.name,
        "crime_file": crime_path.name if crime_path else "Not found (skipped)",
        "listings_loaded": listings_loaded,
        "listings_cleaned": len(listings),
        "calendar_rows_window": calendar_rows_window,
        "reviews_rows_window": reviews_rows_window,
        "crime_rows_window": crime_rows_window,
        "final_shape": str(labeled.shape),
        "listings_with_calendar": listings_with_calendar,
        "listings_with_calendar_pct": (listings_with_calendar / total * 100.0) if total else 0.0,
        "listings_with_reviews": listings_with_reviews,
        "listings_with_reviews_pct": (listings_with_reviews / total * 100.0) if total else 0.0,
        "class_distribution_md": class_distribution_md,
        "top_missing_md": top_missing.to_markdown(),
    }
    write_summary_report(metrics)

    print("\n========== Integration Summary ==========")
    print(f"Files used -> listings: {listings_path.name}, calendar: {calendar_path.name}, reviews: {reviews_path.name}, subway: {subway_path.name}, crime: {metrics['crime_file']}")
    print(f"Listings loaded: {listings_loaded}")
    print(f"Listings after cleaning: {len(listings)}")
    print(f"Calendar rows in window: {calendar_rows_window}")
    print(f"Review rows in window: {reviews_rows_window}")
    print(f"Crime rows in window: {crime_rows_window}")
    print(f"Final table shape: {labeled.shape}")
    print(f"Listings with calendar data: {listings_with_calendar} ({metrics['listings_with_calendar_pct']:.2f}%)")
    print(f"Listings with reviews in window: {listings_with_reviews} ({metrics['listings_with_reviews_pct']:.2f}%)")
    print("Class distribution (consumer_value_class):")
    print(labeled["consumer_value_class"].value_counts(dropna=False).sort_index())
    print("Top 10 columns by missing percentage:")
    print(top_missing)
    print(f"[SUCCESS] Saved final table: {FINAL_TABLE_PATH}")
    print(f"[SUCCESS] Saved data dictionary: {DATA_DICT_PATH}")
    print(f"[SUCCESS] Saved integration summary report: {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
