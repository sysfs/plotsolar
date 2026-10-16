import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.nonparametric.smoothers_lowess import lowess


def read_data(input_file: str | Path, date_format: str | None = None) -> pd.DataFrame:
    """Read a headerless CSV containing date and cumulative value columns."""
    data = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["date", "value"],
        dtype={1: "float64"},
        engine="python",
    )
    data["date"] = pd.to_datetime(data["date"], format=date_format)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.isnull().any().any():
        data = data.dropna()

    if data.empty or len(data) < 2:
        raise ValueError("Not enough data points")

    data = data.sort_values("date").reset_index(drop=True)
    if not (data["value"].diff().dropna() >= 0).all():
        print("[WARN] Values are not strictly increasing — check data integrity")
    return data


def calculate_delta(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate interval rates from validated, chronological observations."""
    rates = data.copy()
    rates["delta_days"] = rates["date"].diff().dt.days
    rates["delta_value"] = rates["value"].diff()
    rates = rates[rates["delta_days"] > 0].copy()
    rates["rate"] = rates["delta_value"] / rates["delta_days"]
    return rates


def build_daily_series(rates: pd.DataFrame) -> pd.DataFrame:
    """Expand each rate interval into daily samples"""
    rows = []
    last_index = len(rates) - 1
    for index, (_, interval) in enumerate(rates.iterrows()):
        days = int(interval["delta_days"])
        start_date = interval["date"] - pd.Timedelta(days=days)
        interval_days = days + (index == last_index)
        for day in range(interval_days):
            rows.append(
                {
                    "date": start_date + pd.Timedelta(days=day),
                    "rate": interval["rate"],
                }
            )

    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No daily data generated")
    return daily


def smooth_data(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Return mean day-of-year rates and their LOWESS-smoothed curve."""
    daily = daily.assign(doy=daily["date"].dt.dayofyear)
    seasonal = daily.groupby("doy")["rate"].mean().reset_index()
    smoothed = lowess(
        seasonal["rate"],
        seasonal["doy"],
        frac=0.1,
        return_sorted=True,
    )
    return seasonal, smoothed


def main(argv: Sequence[str] | None = None) -> int:
    """Solarpanel seasonal output calculator"""
    parser = argparse.ArgumentParser(
        description="Solarpanel seasonal output calculator"
    )
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--date-format", default=None, help="Optional date format")
    args = parser.parse_args(argv)

    try:
        data = read_data(args.input, args.date_format)
    except (OSError, ValueError) as error:
        print(f"[ERROR] Failed to read file: {error}")
        return 1
    print(f"[INFO] read input file {args.input}")

    try:
        data = clean_data(data)
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 2

    rates = calculate_delta(data)
    try:
        daily = build_daily_series(rates)
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 3

    smooth_data(daily)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
