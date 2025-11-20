import pandas as pd

def monthly_income_expense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns monthly total income, expenses, and net.
    """
    df = df.copy()
    
    g = df.groupby(["month", "type"])["amount"].sum()
    unstacked = g.unstack(fill_value=0)

    unstacked["net"] = unstacked.get('income',0) + unstacked.get('expense',0)
    
    return unstacked.reset_index()

def monthly_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pivot table: rows = month, columns = category, values = sum of expenses.
    """
    expenses = df[df["amount"] < 0]

    pivot = expenses.pivot_table(
        index="month",
        columns="category",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )

    return pivot


def top_expenses(df: pd.DataFrame, n=10) -> pd.DataFrame:
    """
    Returns the n largest expense transactions.
    """
    expenses = df[df["amount"] < 0]

    return expenses.sort_values("amount").head(n)
