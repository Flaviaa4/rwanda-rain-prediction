"""Download the raw daily weather data from NASA POWER and assemble the raw dataset.

This is the first stage of the pipeline and is used only by ``01_download_data.ipynb``.
It requests daily values for every configured district centre, stores one CSV per
location so that an interrupted download can resume, removes centres that share a NASA
grid cell with another centre, and writes the combined table and a zip archive that the
student uploads to the public Drive folder.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import ProjectPaths


def fetch_location(name: str, lat: float, lon: float, cfg: dict) -> pd.DataFrame:
    """Download the full daily record for one location from the NASA POWER point API.

    Args:
        name: Location name used to label the rows.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        cfg: The main configuration dictionary (``nasa_power`` section is used).

    Returns:
        A DataFrame with ``date``, ``location``, ``latitude``, ``longitude`` and one column
        per requested parameter, exactly as returned by the API (missing values still coded).

    A single request per location is made for the whole period because the API accepts
    long date ranges, which keeps the number of calls small and polite. Transient failures
    are retried with an increasing pause.
    """
    api = cfg["nasa_power"]
    query = {
        "parameters": ",".join(api["parameters"]),
        "community": api["community"],
        "latitude": lat,
        "longitude": lon,
        "start": api["start"],
        "end": api["end"],
        "format": "JSON",
    }

    # The service occasionally times out or rate-limits, so each failure waits longer
    # than the previous one before the next attempt.
    response = None
    for attempt in range(1, api["max_retries"] + 1):
        try:
            response = requests.get(api["api_url"], params=query,
                                    timeout=api["request_timeout_seconds"])
            if response.status_code == 200:
                break
            print(f"{name}: attempt {attempt} failed with status {response.status_code}: "
                  f"{response.text[:300]}")
        except requests.RequestException as error:
            print(f"{name}: attempt {attempt} raised {error!r}")
        time.sleep(api["request_pause_seconds"] * attempt)
    else:
        raise RuntimeError(f"Could not download {name} after {api['max_retries']} attempts.")

    # The JSON response stores each parameter as a {YYYYMMDD: value} mapping, which pandas
    # turns directly into one column per parameter indexed by date.
    parameter_values = response.json()["properties"]["parameter"]
    frame = pd.DataFrame(parameter_values)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.rename_axis("date").reset_index()
    frame.insert(1, "location", name)
    frame.insert(2, "latitude", lat)
    frame.insert(3, "longitude", lon)
    return frame[["date", "location", "latitude", "longitude", *api["parameters"]]]


def download_all(cfg: dict, paths: ProjectPaths) -> list[Path]:
    """Download every configured location, skipping files that already exist.

    Args:
        cfg: The main configuration dictionary.
        paths: Resolved project paths.

    Returns:
        The list of per-location CSV paths, in configuration order.

    Saving one file per location and skipping existing files means a Colab disconnect in
    the middle of the download costs nothing: re-running the cell continues where it
    stopped.
    """
    api = cfg["nasa_power"]
    files = []
    for name, (lat, lon) in api["locations"].items():
        target = paths.raw_dir / f"{name}.csv"
        files.append(target)
        if target.exists():
            print(f"{name}: already downloaded, skipping")
            continue
        frame = fetch_location(name, lat, lon, cfg)
        frame.to_csv(target, index=False)
        print(f"{name}: saved {len(frame):,} rows")
        # A short pause between requests keeps the load on the public API reasonable.
        time.sleep(api["request_pause_seconds"])
    return files


def drop_duplicate_grid_cells(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove locations whose precipitation series is identical to an earlier location.

    Args:
        data: Combined raw table with ``location``, ``date`` and ``PRECTOTCORR``.

    Returns:
        The filtered table and a mapping ``{dropped_location: kept_location}``.

    NASA POWER returns values for a grid cell of 0.5 x 0.625 degrees rather than for the
    exact coordinates. Two district centres inside the same cell therefore receive
    identical series. Keeping both would double-count the same weather and, worse, the
    model could memorise one copy during training and be tested on its twin, which is a
    form of leakage. Each location's rainfall series is fingerprinted with MD5 and only
    the first location per fingerprint (in configuration order) is kept.
    """
    kept: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for name in pd.unique(data["location"]):
        series = data.loc[data["location"] == name].sort_values("date")["PRECTOTCORR"]
        fingerprint = hashlib.md5(series.to_numpy(dtype=np.float64).tobytes()).hexdigest()
        if fingerprint in kept:
            dropped[name] = kept[fingerprint]
        else:
            kept[fingerprint] = name
    filtered = data.loc[~data["location"].isin(dropped)].reset_index(drop=True)
    return filtered, dropped


def assemble_raw(paths: ProjectPaths, cfg: dict) -> pd.DataFrame:
    """Combine the per-location CSVs into one de-duplicated raw table and save it.

    Args:
        paths: Resolved project paths.
        cfg: The main configuration dictionary.

    Returns:
        The combined raw table with the missing-value code replaced by ``NaN``.

    Files are read in configuration order so that the duplicate check always keeps the
    same centre of each duplicated pair (for example Kigali rather than Muhanga).
    """
    api = cfg["nasa_power"]
    frames = []
    for name in api["locations"]:
        source = paths.raw_dir / f"{name}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing raw file {source}; run notebook 01 first.")
        frames.append(pd.read_csv(source, parse_dates=["date"]))
    data = pd.concat(frames, ignore_index=True)

    # NASA codes missing values as a sentinel number; converting it to NaN prevents the
    # sentinel from being treated as a real (and extreme) measurement.
    data[api["parameters"]] = data[api["parameters"]].replace(api["missing_value"], np.nan)

    data, dropped = drop_duplicate_grid_cells(data)
    for name, twin in dropped.items():
        print(f"Dropped {name}: same NASA grid cell as {twin}")

    data.to_csv(paths.combined_csv, index=False)
    return data


def zip_raw_data(paths: ProjectPaths) -> Path:
    """Write the raw per-location files and the combined table into one zip archive.

    Args:
        paths: Resolved project paths.

    Returns:
        The path of the zip archive.

    The archive is the single file that is uploaded to the public Drive folder so that the
    dataset can be inspected without re-running the download.
    """
    with zipfile.ZipFile(paths.raw_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(paths.raw_dir.glob("*.csv")):
            archive.write(source, arcname=f"raw/{source.name}")
        archive.write(paths.combined_csv, arcname=paths.combined_csv.name)
    return paths.raw_zip


def build_raw_dataset(cfg: dict, paths: ProjectPaths) -> pd.DataFrame:
    """Run the complete download stage and print a short description of the result.

    Args:
        cfg: The main configuration dictionary.
        paths: Resolved project paths.

    Returns:
        The combined, de-duplicated raw table.
    """
    download_all(cfg, paths)
    data = assemble_raw(paths, cfg)
    archive = zip_raw_data(paths)

    # A compact summary lets the reader confirm the download against the expected size
    # (10 locations x 16,436 days) before moving on to the main notebook.
    print(f"\nCombined shape: {data.shape}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print("\nRows per location:")
    print(data["location"].value_counts().sort_index().to_string())
    missing = (data.isna().mean() * 100).sort_values(ascending=False)
    print("\nColumns with the most missing values (%):")
    print(missing[missing > 0].round(2).to_string() or "none")
    print(f"\nSaved {paths.combined_csv}\nSaved {archive}")
    return data
