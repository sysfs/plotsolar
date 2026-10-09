import sys

import pandas as pd

INPUT_FILE = "data.txt"
DATE_FORMAT = None


def main() -> None:
    try:
        df = pd.read_csv(
            INPUT_FILE,
            sep="\t",
            header=None,
            names=["date", "value"],
            dtype={1: "float64"},
            engine="python",
        )
        df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        sys.exit(1)

    print("[INFO] read data.txt")


if __name__ == "__main__":
    main()
