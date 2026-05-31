import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass

matplotlib.use("Agg")  # non-interactive backend — safe for servers

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EDAReport:
    """All computed EDA statistics."""
    shape: tuple
    dtypes: dict
    describe_numeric: dict        # from df.describe()
    describe_categorical: dict    # value counts per cat column
    correlation_matrix: dict      # numeric correlations
    skewness: dict
    saved_figures: list[str]      # paths to saved plots


def run_eda(df: pd.DataFrame, output_prefix: str = "eda") -> EDAReport:
    """
    Run full industry-grade EDA on a cleaned DataFrame.
    Saves figures as PNG files. Returns structured EDAReport.
    """
    saved_figures = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── 1. Describe numeric columns ────────────────────────────────────────
    describe_numeric = (
        df[numeric_cols].describe().round(4).to_dict() if numeric_cols else {}
    )

    # ── 2. Describe categorical columns (top 10 values each) ──────────────
    describe_categorical = {}
    for col in cat_cols:
        describe_categorical[col] = df[col].value_counts().head(10).to_dict()

    # ── 3. Correlation matrix ──────────────────────────────────────────────
    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)
        correlation_matrix = corr.to_dict()
        _save_heatmap(corr, output_prefix, saved_figures)

    # ── 4. Skewness ────────────────────────────────────────────────────────
    skewness = {}
    if numeric_cols:
        skew = df[numeric_cols].skew().round(4)
        skewness = skew.to_dict()

    # ── 5. Missing value heatmap ───────────────────────────────────────────
    if df.isnull().any().any():
        _save_missing_heatmap(df, output_prefix, saved_figures)

    # ── 6. Distribution plots for numeric columns ──────────────────────────
    if numeric_cols:
        _save_distributions(df, numeric_cols, output_prefix, saved_figures)

    # ── 7. Count plots for top categorical columns (max 5) ────────────────
    if cat_cols:
        _save_countplots(df, cat_cols[:5], output_prefix, saved_figures)

    return EDAReport(
        shape=df.shape,
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        describe_numeric=describe_numeric,
        describe_categorical=describe_categorical,
        correlation_matrix=correlation_matrix,
        skewness=skewness,
        saved_figures=saved_figures,
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _save_heatmap(corr: pd.DataFrame, prefix: str, saved: list):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Matrix")
    path = str(FIGURES_DIR / f"{prefix}_correlation.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)


def _save_missing_heatmap(df: pd.DataFrame, prefix: str, saved: list):
    # Only show columns that have at least one missing value
    missing_cols = df.columns[df.isnull().any()].tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        df[missing_cols].isnull(), yticklabels=False,
        cbar=False, cmap="viridis", ax=ax
    )
    ax.set_title("Missing Values (yellow = missing)")
    ax.set_xlabel("Columns")
    path = str(FIGURES_DIR / f"{prefix}_missing.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)


def _save_distributions(df: pd.DataFrame, cols: list, prefix: str, saved: list):
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#378ADD")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Distributions", y=1.02)
    fig.tight_layout()
    path = str(FIGURES_DIR / f"{prefix}_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def _save_countplots(df: pd.DataFrame, cols: list, prefix: str, saved: list):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        top = df[col].value_counts().head(10)
        sns.barplot(x=top.values, y=top.index, ax=axes[i], color="#1D9E75")
        axes[i].set_title(col)
        axes[i].set_xlabel("Count")

    fig.suptitle("Categorical Distributions (Top 10)", y=1.02)
    fig.tight_layout()
    path = str(FIGURES_DIR / f"{prefix}_categoricals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)