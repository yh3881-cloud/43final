from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"

PLOT_DPI = 300
MIN_NEIGHBORHOOD_LISTINGS = 30

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def detect_main_table() -> Path:
    files = [p for p in PROCESSED_DIR.glob("*.csv") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {PROCESSED_DIR}")

    def score(path: Path) -> int:
        n = path.name.lower()
        s = 0
        if "model_table" in n:
            s += 50
        if "undervalued" in n:
            s += 30
        if "processed" in n:
            s += 10
        return s

    files_sorted = sorted(files, key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    chosen = files_sorted[0]
    log(f"Detected main processed table: {chosen}")
    return chosen


def col_exists(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)


def save_fig(fig_num: str, name: str) -> Path:
    out = FIG_DIR / f"{fig_num}_{name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {out}")
    return out


def safe_plot(plot_fn, fig_num: str, fig_name: str) -> Optional[Path]:
    try:
        plot_fn()
        return save_fig(fig_num, fig_name)
    except Exception as exc:
        warn(f"Skipping figure {fig_num}_{fig_name}: {exc}")
        plt.close("all")
        return None


def p99(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(np.nanpercentile(s, 99))


def section_1_dataset_overview(df: pd.DataFrame) -> Dict[str, object]:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    def plot_missing_top20() -> None:
        top20 = missing_pct.head(20).sort_values(ascending=True)
        plt.figure(figsize=(12, 8))
        plt.barh(top20.index, top20.values, color="steelblue")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Column")
        plt.title("Top 20 Columns by Missing Percentage")

    safe_plot(plot_missing_top20, "01", "missing_values_top20")
    return {"missing_pct": missing_pct}


def section_2_price_distribution(df: pd.DataFrame) -> None:
    if "effective_price" not in df.columns:
        warn("`effective_price` missing. Skipping Section 2.")
        return

    df["effective_price"] = pd.to_numeric(df["effective_price"], errors="coerce")
    if "log_effective_price" not in df.columns:
        df["log_effective_price"] = np.log1p(df["effective_price"])
    cap = p99(df["effective_price"])

    def plot_effective_price_dist() -> None:
        plt.figure()
        sns.histplot(df.loc[df["effective_price"] <= cap, "effective_price"].dropna(), bins=60, color="slategray")
        plt.xlabel("Effective Price (capped at 99th percentile)")
        plt.ylabel("Count")
        plt.title("Distribution of Effective Price")

    def plot_log_price_dist() -> None:
        plt.figure()
        sns.histplot(df["log_effective_price"].dropna(), bins=60, color="slategray")
        plt.xlabel("log(1 + Effective Price)")
        plt.ylabel("Count")
        plt.title("Distribution of Log Effective Price")

    def plot_price_by_room_type() -> None:
        if "room_type" not in df.columns:
            raise ValueError("`room_type` missing")
        tmp = df[df["effective_price"] <= cap].copy()
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=tmp, x="room_type", y="effective_price", color="lightsteelblue")
        plt.xlabel("Room Type")
        plt.ylabel("Effective Price (capped at 99th percentile)")
        plt.title("Effective Price by Room Type")
        plt.xticks(rotation=20)

    def plot_price_by_borough() -> None:
        if "neighbourhood_group_cleansed" not in df.columns:
            raise ValueError("`neighbourhood_group_cleansed` missing")
        tmp = df[df["effective_price"] <= cap].copy()
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=tmp, x="neighbourhood_group_cleansed", y="effective_price", color="lightsteelblue")
        plt.xlabel("Borough")
        plt.ylabel("Effective Price (capped at 99th percentile)")
        plt.title("Effective Price by Borough")
        plt.xticks(rotation=20)

    safe_plot(plot_effective_price_dist, "02", "effective_price_distribution")
    safe_plot(plot_log_price_dist, "03", "log_effective_price_distribution")
    safe_plot(plot_price_by_room_type, "04", "price_by_room_type")
    safe_plot(plot_price_by_borough, "05", "price_by_borough")


def section_3_feature_relationships(df: pd.DataFrame) -> None:
    if "effective_price" not in df.columns:
        warn("`effective_price` missing. Skipping Section 3.")
        return
    cap = p99(df["effective_price"])

    def plot_accommodates_vs_price() -> None:
        if "accommodates" not in df.columns:
            raise ValueError("`accommodates` missing")
        plt.figure()
        plt.hexbin(
            pd.to_numeric(df["accommodates"], errors="coerce"),
            pd.to_numeric(df["effective_price"], errors="coerce"),
            gridsize=35,
            cmap="Blues",
            mincnt=1,
        )
        plt.colorbar(label="Count")
        plt.xlabel("Accommodates")
        plt.ylabel("Effective Price")
        plt.title("Accommodates vs Effective Price (Hexbin)")

    def plot_subway_vs_price() -> None:
        if "distance_to_nearest_subway_km" not in df.columns:
            raise ValueError("`distance_to_nearest_subway_km` missing")
        tmp = df[df["effective_price"] <= cap].copy()
        plt.figure()
        plt.scatter(
            pd.to_numeric(tmp["distance_to_nearest_subway_km"], errors="coerce"),
            pd.to_numeric(tmp["effective_price"], errors="coerce"),
            alpha=0.2,
            s=12,
            color="slategray",
        )
        plt.xlabel("Distance to Nearest Subway (km)")
        plt.ylabel("Effective Price (capped at 99th percentile)")
        plt.title("Subway Distance vs Effective Price")

    def plot_rating_vs_price() -> None:
        if "review_scores_rating" not in df.columns:
            raise ValueError("`review_scores_rating` missing")
        tmp = df[df["effective_price"] <= cap].copy()
        plt.figure()
        plt.scatter(
            pd.to_numeric(tmp["review_scores_rating"], errors="coerce"),
            pd.to_numeric(tmp["effective_price"], errors="coerce"),
            alpha=0.2,
            s=12,
            color="slategray",
        )
        plt.xlabel("Review Score Rating")
        plt.ylabel("Effective Price (capped at 99th percentile)")
        plt.title("Rating vs Effective Price")

    def plot_crime_vs_price() -> None:
        if "crime_intensity_log_1000m" not in df.columns:
            raise ValueError("`crime_intensity_log_1000m` missing")
        tmp = df[df["effective_price"] <= cap].copy()
        plt.figure()
        plt.scatter(
            pd.to_numeric(tmp["crime_intensity_log_1000m"], errors="coerce"),
            pd.to_numeric(tmp["effective_price"], errors="coerce"),
            alpha=0.2,
            s=12,
            color="slategray",
        )
        plt.xlabel("Crime Intensity log(1+count within 1000m)")
        plt.ylabel("Effective Price (capped at 99th percentile)")
        plt.title("Crime Intensity vs Effective Price")

    def plot_correlation_heatmap() -> None:
        vars_for_corr = [
            "effective_price", "log_effective_price", "accommodates", "bedrooms", "beds", "bathrooms",
            "amenity_count", "review_scores_rating", "review_scores_value", "review_scores_cleanliness",
            "calendar_available_rate", "calendar_price_volatility", "weekend_price_premium", "reviews_in_window",
            "distance_to_nearest_subway_km", "subway_stations_within_500m", "crime_intensity_log_1000m",
            "host_response_rate", "host_acceptance_rate",
        ]
        existing = [c for c in vars_for_corr if c in df.columns]
        if len(existing) < 2:
            raise ValueError("Not enough numeric columns for correlation heatmap")
        corr = df[existing].apply(pd.to_numeric, errors="coerce").corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap of Key Numeric Features")

    safe_plot(plot_accommodates_vs_price, "06", "accommodates_vs_price")
    safe_plot(plot_subway_vs_price, "07", "subway_distance_vs_price")
    safe_plot(plot_rating_vs_price, "08", "rating_vs_price")
    safe_plot(plot_crime_vs_price, "09", "crime_intensity_vs_price")
    safe_plot(plot_correlation_heatmap, "10", "correlation_heatmap")


def section_4_feature_distributions(df: pd.DataFrame) -> None:
    def hist_plot(column: str, title: str, xlabel: str, fig_num: str, fig_name: str, cap_99: bool = False, log1p: bool = False) -> None:
        if column not in df.columns:
            warn(f"Column `{column}` missing. Skipping figure {fig_num}.")
            return

        def _plot() -> None:
            s = pd.to_numeric(df[column], errors="coerce").dropna()
            if log1p:
                s = np.log1p(s.clip(lower=0))
            if cap_99 and not s.empty:
                s = s[s <= np.nanpercentile(s, 99)]
            plt.figure()
            sns.histplot(s, bins=60, color="slategray")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Count")

        safe_plot(_plot, fig_num, fig_name)

    hist_plot("calendar_available_rate", "Distribution of Calendar Available Rate", "Calendar Available Rate", "11", "calendar_available_rate_distribution")
    hist_plot("calendar_price_volatility", "Distribution of Calendar Price Volatility", "Calendar Price Volatility", "12", "calendar_price_volatility_distribution", cap_99=True)
    hist_plot("reviews_in_window", "Distribution of log(1 + Reviews in Window)", "log(1 + Reviews in Window)", "13", "reviews_in_window_distribution", log1p=True)
    hist_plot("distance_to_nearest_subway_km", "Distribution of Distance to Nearest Subway", "Distance to Nearest Subway (km)", "14", "subway_distance_distribution")
    hist_plot("crime_intensity_log_1000m", "Distribution of Crime Intensity log(1+count)", "Crime Intensity log(1+count within 1000m)", "15", "crime_intensity_distribution")


def build_neighborhood_summary(df: pd.DataFrame) -> pd.DataFrame:
    req = ["neighbourhood_cleansed", "effective_price"]
    if not col_exists(df, req):
        warn("Neighborhood summary skipped: missing required neighborhood or price columns.")
        return pd.DataFrame()

    group_cols = ["neighbourhood_cleansed"]
    if "neighbourhood_group_cleansed" in df.columns:
        group_cols.append("neighbourhood_group_cleansed")

    agg_map = {
        "id": "count",
        "effective_price": "median",
        "review_scores_rating": "median",
        "distance_to_nearest_subway_km": "median",
        "calendar_available_rate": "median",
        "reviews_in_window": "median",
        "crime_intensity_log_1000m": "median",
    }
    valid_agg = {k: v for k, v in agg_map.items() if k in df.columns}
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(valid_agg)
        .rename(
            columns={
                "id": "listing_count",
                "effective_price": "median_effective_price",
                "review_scores_rating": "median_rating",
                "distance_to_nearest_subway_km": "median_subway_distance",
                "calendar_available_rate": "median_calendar_available_rate",
                "reviews_in_window": "median_reviews_in_window",
                "crime_intensity_log_1000m": "median_crime_intensity",
            }
        )
        .reset_index()
    )
    out = out[out["listing_count"] >= MIN_NEIGHBORHOOD_LISTINGS].copy()
    return out


def section_5_neighborhood(df: pd.DataFrame) -> None:
    nbh = build_neighborhood_summary(df)
    if nbh.empty:
        return

    def plot_top15_by_price() -> None:
        tmp = nbh.sort_values("median_effective_price", ascending=False).head(15).iloc[::-1]
        plt.figure(figsize=(12, 8))
        plt.barh(tmp["neighbourhood_cleansed"], tmp["median_effective_price"], color="steelblue")
        plt.xlabel("Median Effective Price")
        plt.ylabel("Neighborhood")
        plt.title("Top 15 Neighborhoods by Median Effective Price")

    def plot_top15_by_count() -> None:
        tmp = nbh.sort_values("listing_count", ascending=False).head(15).iloc[::-1]
        plt.figure(figsize=(12, 8))
        plt.barh(tmp["neighbourhood_cleansed"], tmp["listing_count"], color="steelblue")
        plt.xlabel("Listing Count")
        plt.ylabel("Neighborhood")
        plt.title("Top 15 Neighborhoods by Listing Count")

    def plot_price_vs_subway() -> None:
        if "median_subway_distance" not in nbh.columns:
            raise ValueError("median_subway_distance missing")
        plt.figure(figsize=(10, 7))
        hue = "neighbourhood_group_cleansed" if "neighbourhood_group_cleansed" in nbh.columns else None
        sns.scatterplot(
            data=nbh,
            x="median_subway_distance",
            y="median_effective_price",
            size="listing_count",
            hue=hue,
            alpha=0.75,
            palette="deep",
        )
        plt.xlabel("Median Subway Distance (km)")
        plt.ylabel("Median Effective Price")
        plt.title("Neighborhood Median Price vs Subway Distance")

    def plot_price_vs_crime() -> None:
        if "median_crime_intensity" not in nbh.columns:
            raise ValueError("median_crime_intensity missing")
        plt.figure(figsize=(10, 7))
        hue = "neighbourhood_group_cleansed" if "neighbourhood_group_cleansed" in nbh.columns else None
        sns.scatterplot(
            data=nbh,
            x="median_crime_intensity",
            y="median_effective_price",
            size="listing_count",
            hue=hue,
            alpha=0.75,
            palette="deep",
        )
        plt.xlabel("Median Crime Intensity")
        plt.ylabel("Median Effective Price")
        plt.title("Neighborhood Median Price vs Crime Intensity")

    safe_plot(plot_top15_by_price, "16", "top_neighborhoods_by_price")
    safe_plot(plot_top15_by_count, "17", "top_neighborhoods_by_listing_count")
    safe_plot(plot_price_vs_subway, "18", "neighborhood_price_vs_subway")
    safe_plot(plot_price_vs_crime, "19", "neighborhood_price_vs_crime")


def main() -> None:
    ensure_dirs()
    table_path = detect_main_table()
    df = pd.read_csv(table_path, low_memory=False)

    if "effective_price" in df.columns and "log_effective_price" not in df.columns:
        df["log_effective_price"] = np.log1p(pd.to_numeric(df["effective_price"], errors="coerce"))
        log("Created missing `log_effective_price` from `effective_price`.")

    log("Running EDA Pre script: figures 01-19 only.")
    section_1_dataset_overview(df)
    section_2_price_distribution(df)
    section_3_feature_relationships(df)
    section_4_feature_distributions(df)
    section_5_neighborhood(df)
    log("EDA pre figures completed.")


if __name__ == "__main__":
    main()
