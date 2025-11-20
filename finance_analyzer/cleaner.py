# finance_analyzer/cleaner.py
import pandas as pd

def parse_pl_number(col: pd.Series) -> pd.Series:
    """
    Convert Polish-formatted numbers like '1 234,56 zł' -> 1234.56 (float).

    - Handles normal + non-breaking spaces
    - Handles comma as decimal separator
    - Strips non-numeric junk (currency symbols etc.)
    """
    s = (
        col.astype(str)
           .str.replace("\u00a0", "", regex=False)   # non-breaking space
           .str.replace(" ", "", regex=False)        # normal spaces
           .str.replace(",", ".", regex=False)       # decimal comma -> dot
           .str.replace(r"[^\d\.-]", "", regex=True) # keep digits, dot, minus
    )

    # Let pandas handle NA + conversion
    return pd.to_numeric(s, errors="coerce")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 0) Drop obviously empty "Unnamed" columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.dropna(axis=1, how="all")

    # 1) Build a flexible column mapping: handle both good and mangled Polish letters
    cols_map: dict[str, str] = {}

    def add_mapping(possible_names, standard_name):
        """If any of possible_names is present in df.columns, map it."""
        for name in possible_names:
            if name in df.columns:
                cols_map[name] = standard_name
                break  # stop after first match

    add_mapping(["Data transakcji"], "date")
    add_mapping(["Data księgowania", "Data ksi�gowania"], "posting_date")
    add_mapping(["Dane kontrahenta"], "counterparty")
    add_mapping(["Tytuł", "Tytu�"], "title")
    add_mapping(["Szczegóły", "Szczeg�y"], "details")
    add_mapping(["Kwota transakcji (waluta rachunku)"], "amount_raw")
    add_mapping(["Saldo po transakcji"], "balance_raw")
    add_mapping(["Waluta"], "currency")

    # Keep only columns we know how to interpret
    df = df[[c for c in df.columns if c in cols_map.keys()]]
    df = df.rename(columns=cols_map)

    # 2) Build a human-readable 'description' from whatever we have
    desc_parts = [c for c in ["counterparty", "title", "details"] if c in df.columns]
    if desc_parts:
        df["description"] = df[desc_parts].astype(str).agg(" | ".join, axis=1)
    else:
        df["description"] = ""

    # 3) Dates (Polish bank exports are usually day-first)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if "posting_date" in df.columns:
        df["posting_date"] = pd.to_datetime(df["posting_date"], dayfirst=True, errors="coerce")

    # 4) Numeric columns
    if "amount_raw" in df.columns:
        df["amount"] = parse_pl_number(df["amount_raw"])
    else:
        df["amount"] = pd.NA

    if "balance_raw" in df.columns:
        df["balance"] = parse_pl_number(df["balance_raw"])
    else:
        df["balance"] = pd.NA

    # 5) Type: income vs expense (only if amount is numeric)
    df["type"] = df["amount"].apply(lambda x: "income" if pd.notna(x) and x > 0 else "expense")

    # 6) Time helper columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")

    # 7) Final column order (only keep what exists)
    keep_cols = ["date", "posting_date", "description",
                 "amount", "balance", "currency",
                 "type", "year", "month"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # 8) Sort
    df = df.sort_values("date").reset_index(drop=True)

    return df
