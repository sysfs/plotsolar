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


def main(argv: Sequence[str] | None = None) -> int:
    """Solarpanel seasonal output calculator"""
    parser = argparse.ArgumentParser(
        description="Solarpanel seasonal output calculator"
    )
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--date-format", default=None, help="Optional date format")
    args = parser.parse_args(argv)

    try:
        read_data(args.input, args.date_format)
    except (OSError, ValueError) as error:
        print(f"[ERROR] Failed to read file: {error}")
        return 1

    print(f"[INFO] read input file {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
