# finance_analyzer/recurring.py

import pandas as pd
import numpy as np
from typing import Optional


def _normalize_description(desc: pd.Series) -> pd.Series:
    """
    Normalize transaction descriptions for grouping:
    - lowercase
    - strip spaces
    - remove numbers (often reference/IDs)
    - collapse multiple spaces
    """
    s = (
        desc.fillna("")
            .str.lower()
            .str.replace(r"\d+", "", regex=True)      # remove digits
            .str.replace(r"\s+", " ", regex=True)     # collapse spaces
            .str.strip()
    )
    return s


def find_recurring_transactions(
    df: pd.DataFrame,
    min_occurrences: int = 3,
    target_interval_days: int = 30,
    interval_tolerance: int = 4,
) -> pd.DataFrame:
    """
    Heuristic detector of recurring payments (subscriptions, rent, etc.).

    Rules:
      - only consider EXPENSES (amount < 0)
      - group by normalized description
      - for each group:
          * must have at least `min_occurrences`
          * compute diffs between consecutive dates (in days)
          * mean(diff) close to `target_interval_days` (+/- interval_tolerance)
    
    Returns a DataFrame with:
      desc_norm, example_description, n_payments, avg_amount, std_amount,
      mean_interval_days, std_interval_days, first_date, last_date, category
    """

    if df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid modifying the original
    data = df.copy()

    # 1) Only expenses
    expenses = data[data["amount"] < 0].copy()
    if expenses.empty:
        return pd.DataFrame()

    # 2) Normalized description for grouping
    expenses["desc_norm"] = _normalize_description(expenses["description"])

    # 3) For each normalized description, analyze payment pattern
    rows = []

    for desc_norm, group in expenses.groupby("desc_norm"):
        group = group.sort_values("date")
        if group.shape[0] < min_occurrences:
            continue

        # date differences in days between consecutive payments
        diffs = group["date"].diff().dt.days.dropna()
        if diffs.empty:
            continue

        mean_interval = diffs.mean()
        std_interval = diffs.std(ddof=0) if len(diffs) > 1 else 0.0

        # Heuristic: about once a month
        if not (
            (target_interval_days - interval_tolerance)
            <= mean_interval
            <= (target_interval_days + interval_tolerance)
        ):
            continue  # not regular enough

        # Aggregate amounts
        amounts = group["amount"]
        avg_amount = amounts.mean()
        std_amount = amounts.std(ddof=0) if len(amounts) > 1 else 0.0

        # Example: one "representative" description and category
        example_desc = group["description"].iloc[0]
        example_cat: Optional[str] = None
        if "category" in group.columns:
            example_cat = group["category"].mode().iloc[0]

        rows.append(
            {
                "desc_norm": desc_norm,
                "example_description": example_desc,
                "category": example_cat,
                "n_payments": int(group.shape[0]),
                "avg_amount": float(avg_amount),
                "std_amount": float(std_amount),
                "mean_interval_days": float(mean_interval),
                "std_interval_days": float(std_interval),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
            }
        )

    result = pd.DataFrame(rows)

    # Sort: most frequent / most recent first
    if not result.empty:
        result = result.sort_values(
            by=["n_payments", "last_date"], ascending=[False, False]
        ).reset_index(drop=True)

    return result
