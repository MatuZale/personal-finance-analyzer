import pandas as pd
import json
from pathlib import Path

def load_category_rules(path: str|Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def categorize_transactions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    df = df.copy()
    df["category"] = "Uncategorized"

    desc = df["description"].str.lower().fillna("")

    for category, keywords in rules.items():
        mask = pd.Series(False, index=df.index)
        for kw in keywords:
            mask |=desc.str.contains(kw)
        df.loc[mask, "category"] = category

    return df

def amount_manual_overrides(df: pd.DataFrame, path: str|Path) -> pd.DataFrame:
    ov = pd.read_csv(path)
    desc = df["description"].str.lower().fillna("")
    for _, row in ov.iterrows():
        mask = desc.str.contains(str(row["substring"]).lower())
        df.loc[mask, "category"] = row["category"]
    return df