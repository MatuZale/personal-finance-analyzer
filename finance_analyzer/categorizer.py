# finance_analyzer/categorizer.py

import pandas as pd
import json
from pathlib import Path


# ------------------ RULE-BASED CATEGORIES ------------------ #

def load_category_rules(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_transactions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Apply rule-based categorization using description substrings.

    Parameters
    ----------
    df : DataFrame
        Cleaned transactions, must have 'description' column.
    rules : dict
        Mapping: category -> list of keywords (lowercase substrings).

    Returns
    -------
    DataFrame with a new/updated 'category' column.
    """
    df = df.copy()
    # Initialize with Uncategorized if not present
    if "category" not in df.columns:
        df["category"] = "Uncategorized"

    desc = df["description"].astype(str).str.lower().fillna("")

    for category, keywords in rules.items():
        mask = pd.Series(False, index=df.index)
        for kw in keywords:
            kw = str(kw).lower()
            if not kw:
                continue
            mask |= desc.str.contains(kw)
        df.loc[mask, "category"] = category

    return df


# ------------------ MANUAL OVERRIDES ------------------ #

def load_manual_overrides(path: str | Path) -> pd.DataFrame:
    """
    Load manual overrides from a CSV like:

        substring,category
        NETFLIX,Entertainment
        BOLT,Transport

    Returns empty DataFrame if file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["substring", "category"])

    try:
        df = pd.read_csv(path)
    except Exception:
        # Fallback to empty if format is broken
        return pd.DataFrame(columns=["substring", "category"])

    # Ensure correct columns
    expected_cols = {"substring", "category"}
    if not expected_cols.issubset(set(df.columns)):
        return pd.DataFrame(columns=["substring", "category"])

    # Drop completely empty rows
    df = df.dropna(subset=["substring", "category"])
    df["substring"] = df["substring"].astype(str)
    df["category"] = df["category"].astype(str)
    return df.reset_index(drop=True)


def apply_manual_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """
    Apply manual overrides on top of existing categories.

    For each row in `overrides`:
      - substring: a piece of text to search in description
      - category: category name to assign

    Any matching transaction (description contains substring) will be
    reassigned to the given category. Manual overrides take precedence
    over rule-based ones.
    """
    df = df.copy()
    if overrides is None or overrides.empty:
        return df

    if "description" not in df.columns:
        return df

    desc = df["description"].astype(str).str.lower().fillna("")

    for _, row in overrides.iterrows():
        substr = str(row["substring"]).strip()
        cat = str(row["category"]).strip()
        if not substr or not cat:
            continue
        substr_low = substr.lower()
        mask = desc.str.contains(substr_low)
        df.loc[mask, "category"] = cat

    return df
