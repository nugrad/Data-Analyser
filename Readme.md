# EDA Tool

Upload a messy CSV or Excel file and get back a cleaned dataset, a data quality report, and saved visualizations.

---

## Setup

```bash
git clone https://github.com/nugrad/Data-Analyser.git
cd eda_tool

python -m venv venv
venv/Scripts/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Project structure

```
eda_tool/
├── data/
│   ├── raw/              ← place your input files here
│   └── processed/        ← cleaned outputs land here
├── src/
│   └── eda_tool/
│       ├── loader.py     ← reads CSV / Excel
│       ├── cleaner.py    ← data quality checks
│       └── analyzer.py   ← EDA and plot generation
├── reports/
│   └── figures/          ← saved PNG plots
├── api/
│   └── main.py           ← FastAPI endpoint
└── requirements.txt
```

---

## How to run

### Option 1 — Python script

Drop your file into `data/raw/`, then:

```python
from src.eda_tool.loader import load_file
from src.eda_tool.cleaner import run_cleaning_checks
from src.eda_tool.analyzer import run_eda

df = load_file("data/raw/your_file.csv")
df_clean, cleaning_report = run_cleaning_checks(df)
eda_report = run_eda(df_clean, output_prefix="your_file")

df_clean.to_csv("data/processed/your_file_cleaned.csv", index=False)
```

### Option 2 — API

```bash
uvicorn api.main:app --reload
```

Upload a file:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@data/raw/your_file.csv"
```

Download the cleaned file:

```bash
curl http://localhost:8000/download/your_file_cleaned.csv -o cleaned.csv
```

API docs at `http://localhost:8000/docs`.

---

## What it does

**Step 1 — Load**

Reads CSV or Excel. Tries multiple encodings automatically so files with special characters don't fail silently.

**Step 2 — Clean**

| Check | What happens |
|-------|-------------|
| Column names | Lowercased, spaces replaced with underscores |
| Empty columns | Dropped |
| Duplicate rows | Removed |
| Missing values | Counted and reported per column |
| Dtypes | Numeric and datetime columns inferred from strings |
| Constant columns | Flagged — zero variance means no analytical value |
| High cardinality | Flagged — likely IDs or free text |
| Outliers | Detected per numeric column using the IQR method |

**Step 3 — Analyze**

Generates and saves to `reports/figures/`:

- Correlation heatmap
- Distribution plots for every numeric column
- Count plots for categorical columns
- Missing value heatmap (if applicable)

---

## Supported file types

`.csv` `.xlsx` `.xls`

---

## Requirements

```
fastapi
uvicorn[standard]
pandas
openpyxl
matplotlib
seaborn
python-multipart
```