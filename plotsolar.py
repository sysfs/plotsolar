from pathlib import Path

import pandas as pd

INPUT_FILE = "data.txt"
DATE_FORMAT = None


def read_data(input_file: str | Path, date_format: str | None = None) -> pd.DataFrame:
    """Read a headerless TSV containing date and cumulative value columns."""
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


def main() -> int:
    try:
        read_data(INPUT_FILE, DATE_FORMAT)
    except (OSError, ValueError) as error:
        print(f"[ERROR] Failed to read file: {error}")
        return 1

    print("[INFO] read data.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
