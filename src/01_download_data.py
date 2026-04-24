from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


# =========================
# Configuration
# =========================
LISTINGS_SNAPSHOT_DATE = "2025-04-01"
CALENDAR_SNAPSHOT_DATE = "2025-04-01"
REVIEWS_SNAPSHOT_DATE = "2026-04-01"

INSIDE_AIRBNB_BASE = "https://data.insideairbnb.com/united-states/ny/new-york-city"
INSIDE_AIRBNB_GET_DATA_PAGE = "https://insideairbnb.com/get-the-data/"
MTA_SUBWAY_CSV_URL = "https://data.ny.gov/resource/39hk-dx4f.csv?$limit=50000"
NYPD_HISTORIC_API = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
NYPD_YTD_API = "https://data.cityofnewyork.us/resource/5uac-w243.json"
RESEARCH_START = "2025-04-01"
RESEARCH_END = "2026-03-31"
NYPD_PAGE_SIZE = 50000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MB
REQUEST_TIMEOUT_SECONDS = 60


def build_inside_airbnb_url(snapshot_date: str, filename: str) -> str:
    return f"{INSIDE_AIRBNB_BASE}/{snapshot_date}/data/{filename}"


def get_available_snapshot_dates(filename: str) -> List[str]:
    """
    Scrape Inside Airbnb 'Get the Data' page and return available NYC snapshot
    dates for a specific file (e.g. listings.csv.gz, calendar.csv.gz, reviews.csv.gz).
    """
    print(f"[INFO] Fetching available snapshot dates for {filename} from Inside Airbnb...")
    try:
        response = requests.get(
            INSIDE_AIRBNB_GET_DATA_PAGE, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARNING] Could not fetch Get the Data page: {exc}")
        return []

    # Match NYC links such as:
    # https://data.insideairbnb.com/united-states/ny/new-york-city/YYYY-MM-DD/data/listings.csv.gz
    pattern = (
        r"https://data\.insideairbnb\.com/united-states/ny/new-york-city/"
        r"(\d{4}-\d{2}-\d{2})/data/"
        + re.escape(filename)
    )
    matches = re.findall(pattern, response.text)
    dates = sorted(set(matches))
    print(f"[INFO] Found {len(dates)} available NYC snapshot dates for {filename}.")
    return dates


def choose_fallback_snapshot_date(target_date: str, available_dates: List[str]) -> Optional[str]:
    if not available_dates:
        return None
    if target_date in available_dates:
        return target_date

    target_ts = int(target_date.replace("-", ""))
    # Pick nearest by absolute calendar distance in YYYYMMDD numeric space.
    best = min(available_dates, key=lambda d: abs(int(d.replace("-", "")) - target_ts))
    return best


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured raw data directory exists: {RAW_DIR}")


def fetch_socrata_rows(
    endpoint: str,
    where_clause: str,
    columns: List[str],
    source_name: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    offset = 0

    print(f"[INFO] Pulling {source_name} from Socrata API...")
    while True:
        params = {
            "$select": ",".join(columns),
            "$where": where_clause,
            "$limit": NYPD_PAGE_SIZE,
            "$offset": offset,
            "$order": "cmplnt_fr_dt ASC",
        }
        try:
            response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            batch = response.json()
        except requests.RequestException as exc:
            print(f"[ERROR] Failed requesting {source_name}: {exc}")
            break
        except ValueError as exc:
            print(f"[ERROR] Failed parsing {source_name} JSON response: {exc}")
            break

        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)
        print(f"[INFO] {source_name}: fetched {len(rows)} rows so far...")

        if len(batch) < NYPD_PAGE_SIZE:
            break

    return rows


def save_rows_to_csv(rows: List[Dict[str, str]], columns: List[str], destination: Path) -> bool:
    if not rows:
        print(f"[ERROR] No rows to save for {destination.name}.")
        return False
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination.exists() and destination.stat().st_size > 0


def download_nypd_complaints(destination: Path) -> bool:
    useful_columns = [
        "cmplnt_num",
        "cmplnt_fr_dt",
        "cmplnt_fr_tm",
        "ofns_desc",
        "law_cat_cd",
        "boro_nm",
        "latitude",
        "longitude",
    ]
    where_clause = (
        f"cmplnt_fr_dt between '{RESEARCH_START}T00:00:00' and "
        f"'{RESEARCH_END}T23:59:59'"
    )

    historic_rows = fetch_socrata_rows(
        endpoint=NYPD_HISTORIC_API,
        where_clause=where_clause,
        columns=useful_columns,
        source_name="NYPD Complaint Historic",
    )
    ytd_rows = fetch_socrata_rows(
        endpoint=NYPD_YTD_API,
        where_clause=where_clause,
        columns=useful_columns,
        source_name="NYPD Complaint Current Year To Date",
    )

    combined_rows = historic_rows + ytd_rows
    if not combined_rows:
        print(
            "[ERROR] NYPD complaint download returned zero rows. "
            "Please verify NYC Open Data API availability and date filters."
        )
        return False

    # Keep one row per complaint id to avoid overlap between datasets.
    deduped: Dict[str, Dict[str, str]] = {}
    for row in combined_rows:
        cmplnt_num = row.get("cmplnt_num")
        key = cmplnt_num if cmplnt_num else f"row_{len(deduped)}"
        deduped[key] = row

    ok = save_rows_to_csv(list(deduped.values()), useful_columns, destination)
    if not ok:
        print(f"[ERROR] Failed to save NYPD complaint data: {destination}")
        return False

    print(
        f"[SUCCESS] Saved NYPD complaint rows: {len(deduped)} -> {destination} "
        f"({format_file_size(destination.stat().st_size)})"
    )
    return True


def get_expected_size(response: requests.Response) -> Optional[int]:
    content_length = response.headers.get("Content-Length")
    if not content_length:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def format_file_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def stream_download(url: str, destination: Path, source_name: str) -> bool:
    print(f"\n[INFO] Downloading {source_name}")
    print(f"[INFO] URL: {url}")

    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status_code == 404:
                print(
                    f"[ERROR] URL returned 404 for {source_name}.\n"
                    f"        Requested URL: {url}\n"
                    f"        Please check the closest available NYC snapshot on {INSIDE_AIRBNB_GET_DATA_PAGE}\n"
                    f"        and update the snapshot date constant. If needed, request archived data."
                )
                return False

            response.raise_for_status()
            expected_size = get_expected_size(response)

            progress = tqdm(
                total=expected_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {destination.name}",
            )
            with destination.open("wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            progress.close()

    except requests.RequestException as exc:
        print(f"[ERROR] Failed to download {source_name}: {exc}")
        return False

    if not destination.exists():
        print(f"[ERROR] Download completed but file not found: {destination}")
        return False

    downloaded_size = destination.stat().st_size
    if downloaded_size == 0:
        print(f"[ERROR] Downloaded file is empty: {destination}")
        return False

    print(
        f"[SUCCESS] Saved {source_name} -> {destination} "
        f"({format_file_size(downloaded_size)})"
    )
    return True


def download_inside_airbnb_with_fallback(
    target_date: str,
    filename: str,
    destination: Path,
    source_name: str,
) -> bool:
    primary_url = build_inside_airbnb_url(target_date, filename)
    primary_ok = stream_download(primary_url, destination, source_name)
    if primary_ok:
        return True

    # Try fallback by scraping available dates.
    available_dates = get_available_snapshot_dates(filename)
    fallback_date = choose_fallback_snapshot_date(target_date, available_dates)

    if not fallback_date:
        print(
            f"[ERROR] Could not find fallback snapshot date for {filename}. "
            f"Please manually check {INSIDE_AIRBNB_GET_DATA_PAGE} and update constants."
        )
        return False

    if fallback_date == target_date:
        # If date is same, failure likely due to network permission/connectivity.
        print(
            f"[ERROR] Target date {target_date} appears in listing, but download failed. "
            "Please check network access and retry."
        )
        return False

    print(
        f"[WARNING] Snapshot {target_date} for {filename} not directly downloadable. "
        f"Using nearest available snapshot: {fallback_date}"
    )
    fallback_url = build_inside_airbnb_url(fallback_date, filename)
    fallback_ok = stream_download(
        fallback_url,
        destination,
        f"{source_name} (fallback snapshot {fallback_date})",
    )
    if fallback_ok:
        print(
            f"[INFO] Saved fallback data to {destination.name}; requested={target_date}, "
            f"actual={fallback_date}"
        )
    return fallback_ok


def main() -> None:
    print("[INFO] Starting data download pipeline...")
    ensure_directories()

    all_ok = True
    all_ok = all_ok and download_inside_airbnb_with_fallback(
        target_date=LISTINGS_SNAPSHOT_DATE,
        filename="listings.csv.gz",
        destination=RAW_DIR / f"listings_{LISTINGS_SNAPSHOT_DATE}.csv.gz",
        source_name="Inside Airbnb NYC detailed listings",
    )
    all_ok = all_ok and download_inside_airbnb_with_fallback(
        target_date=CALENDAR_SNAPSHOT_DATE,
        filename="calendar.csv.gz",
        destination=RAW_DIR / f"calendar_{CALENDAR_SNAPSHOT_DATE}.csv.gz",
        source_name="Inside Airbnb NYC detailed calendar",
    )
    all_ok = all_ok and download_inside_airbnb_with_fallback(
        target_date=REVIEWS_SNAPSHOT_DATE,
        filename="reviews.csv.gz",
        destination=RAW_DIR / f"reviews_{REVIEWS_SNAPSHOT_DATE}.csv.gz",
        source_name="Inside Airbnb NYC detailed reviews",
    )
    all_ok = all_ok and stream_download(
        url=MTA_SUBWAY_CSV_URL,
        destination=RAW_DIR / "mta_subway_stations.csv",
        source_name="MTA Subway Stations",
    )
    all_ok = all_ok and download_nypd_complaints(
        destination=RAW_DIR / f"nypd_complaints_{RESEARCH_START}_{RESEARCH_END}.csv"
    )

    if all_ok:
        print("\n[SUCCESS] All raw files downloaded successfully.")
    else:
        print(
            "\n[WARNING] Some downloads failed. Fix the reported issues and rerun "
            "this script."
        )


if __name__ == "__main__":
    main()
