import pandas as pd
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleaningReport:
    """Structured report of all data quality issues found."""
    shape_before: tuple
    shape_after: tuple
    missing_summary: dict[str, Any]
    duplicate_rows_removed: int
    columns_dropped: list[str]
    dtypes_inferred: dict[str, str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    outlier_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def run_cleaning_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Perform industry-standard cleaning checks and return a cleaned DataFrame
    alongside a structured report.

    This does NOT make destructive changes silently — it documents everything.
    """
    warnings = []
    columns_dropped = []
    shape_before = df.shape

    # ── 1. Strip whitespace from column names ──────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── 2. Drop fully empty rows and columns ──────────────────────────────
    cols_all_null = df.columns[df.isnull().all()].tolist()
    if cols_all_null:
        df.drop(columns=cols_all_null, inplace=True)
        columns_dropped.extend(cols_all_null)
        warnings.append(f"Dropped {len(cols_all_null)} fully-empty column(s): {cols_all_null}")

    df.dropna(how="all", inplace=True)

    # ── 3. Duplicate rows ──────────────────────────────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    duplicate_rows_removed = n_before - len(df)
    if duplicate_rows_removed:
        warnings.append(f"Removed {duplicate_rows_removed} duplicate row(s).")

    # ── 4. Missing value summary ───────────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_summary = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing_counts[col] > 0
    }

    # Warn on columns with >50% missing
    for col, stats in missing_summary.items():
        if stats["pct"] > 50:
            warnings.append(
                f"Column '{col}' is {stats['pct']}% missing — consider dropping it."
            )

    # ── 5. Infer better dtypes ─────────────────────────────────────────────
    # pandas read_csv often reads numeric columns as object when mixed with nulls
    dtypes_inferred = {}
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / len(df) > 0.8:  # 80%+ successfully converted
            df[col] = converted
            dtypes_inferred[col] = "numeric (inferred)"

    # Try datetime inference on object columns with "date" or "time" in name
    for col in df.select_dtypes(include="object").columns:
        if any(kw in col for kw in ["date", "time", "dt", "year"]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                dtypes_inferred[col] = "datetime (inferred)"
            except Exception:
                pass

    # ── 6. Constant columns (zero variance) ───────────────────────────────
    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        warnings.append(
            f"Constant columns (no variance): {constant_columns}. "
            "Consider dropping them."
        )

    # ── 7. High cardinality categoricals ──────────────────────────────────
    # These are likely IDs or free-text — flag, don't drop automatically
    high_cardinality_columns = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique() / len(df) > 0.9
    ]
    if high_cardinality_columns:
        warnings.append(
            f"High-cardinality columns (likely IDs or free text): {high_cardinality_columns}"
        )

    # ── 8. Outlier detection (IQR method on numeric columns) ───────────────
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())
        if n_outliers > 0:
            outlier_summary[col] = {
                "count": n_outliers,
                "pct": round(n_outliers / len(df) * 100, 2),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
            }

    report = CleaningReport(
        shape_before=shape_before,
        shape_after=df.shape,
        missing_summary=missing_summary,
        duplicate_rows_removed=duplicate_rows_removed,
        columns_dropped=columns_dropped,
        dtypes_inferred=dtypes_inferred,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        outlier_summary=outlier_summary,
        warnings=warnings,
    )

    return df, report