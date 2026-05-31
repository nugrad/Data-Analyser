import pandas as pd
from pathlib import Path


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_file(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.
    Raises ValueError for unsupported file types.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".csv":
        # Try common encodings — real-world files are messy
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the CSV file with common encodings.")

    # Excel
    df = pd.read_excel(filepath, engine="openpyxl")
    return df