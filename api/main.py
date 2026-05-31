"""
EDA Tool API
------------
POST /analyze  →  Upload a CSV or Excel file, get back a full EDA + cleaning report.
GET  /health   →  Liveness check.
"""

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from src.eda_tool.loader import load_file
from src.eda_tool.cleaner import run_cleaning_checks
from src.eda_tool.analyzer import run_eda


app = FastAPI(
    title="EDA Tool",
    description="Upload messy CSV/Excel → get a cleaned file + full EDA report.",
    version="1.0.0",
)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main endpoint. Accepts a CSV or Excel file.
    Returns: cleaning report, EDA stats, paths to saved figures.
    """
    allowed_ext = {".csv", ".xlsx", ".xls"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload CSV or Excel only."
        )

    # Save upload to a temp file so our loader can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # ── Step 1: Load ───────────────────────────────────────────────────
        df = load_file(tmp_path)

        # ── Step 2: Clean ──────────────────────────────────────────────────
        df_clean, cleaning_report = run_cleaning_checks(df)

        # ── Step 3: EDA ────────────────────────────────────────────────────
        prefix = Path(file.filename).stem
        eda_report = run_eda(df_clean, output_prefix=prefix)

        # ── Step 4: Save cleaned file ──────────────────────────────────────
        clean_path = PROCESSED_DIR / f"{prefix}_cleaned.csv"
        df_clean.to_csv(clean_path, index=False)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # always clean up temp file

    return JSONResponse({
        "filename": file.filename,
        "cleaning_report": {
            "shape_before": cleaning_report.shape_before,
            "shape_after": cleaning_report.shape_after,
            "duplicate_rows_removed": cleaning_report.duplicate_rows_removed,
            "columns_dropped": cleaning_report.columns_dropped,
            "missing_summary": cleaning_report.missing_summary,
            "dtypes_inferred": cleaning_report.dtypes_inferred,
            "constant_columns": cleaning_report.constant_columns,
            "high_cardinality_columns": cleaning_report.high_cardinality_columns,
            "outlier_summary": cleaning_report.outlier_summary,
            "warnings": cleaning_report.warnings,
        },
        "eda_report": {
            "shape": eda_report.shape,
            "dtypes": eda_report.dtypes,
            "describe_numeric": eda_report.describe_numeric,
            "describe_categorical": eda_report.describe_categorical,
            "skewness": eda_report.skewness,
            "saved_figures": eda_report.saved_figures,
        },
        "cleaned_file": str(clean_path),
    })


@app.get("/download/{filename}")
def download_cleaned(filename: str):
    """Download a previously cleaned file by name."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="text/csv", filename=filename)