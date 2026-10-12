import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
