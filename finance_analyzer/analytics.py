from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np


def compute_monthly_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liczy miesięczny cashflow na podstawie pojedynczych transakcji.

    Wejście:
        df: DataFrame z kolumnami:
            - 'date'   : datetime64[ns] (albo coś, co da się przekonwertować na datę)
            - 'amount' : liczba (dodatnia = wpływ, ujemna = wydatek)

    Wyjście:
        DataFrame z kolumnami:
            - 'month'   : pierwszy dzień miesiąca (np. 2025-01-01)
            - 'income'  : suma wpływów w danym miesiącu (>= 0)
            - 'expense' : suma wydatków w danym miesiącu (<= 0)
            - 'net'     : income + expense (czyli saldo netto miesiąca)
    """
    df = df.copy()

    # Upewniamy się, że mamy daty
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # wyrzucamy wiersze bez daty
    df = df.dropna(subset=["date"])

    # Normalizujemy do "miesiąca" — tutaj: pierwszy dzień miesiąca
    df["month"] = df["date"].values.astype("datetime64[M]")

    # Grupujemy po miesiącu
    grouped = df.groupby("month")["amount"]

    # Osobno wpływy i wydatki
    income = grouped.apply(lambda s: s[s > 0].sum()).rename("income")
    expense = grouped.apply(lambda s: s[s < 0].sum()).rename("expense")

    out = pd.concat([income, expense], axis=1).fillna(0.0)
    out["net"] = out["income"] + out["expense"]

    # Wyjście posortowane po czasie, z kolumną 'month' jako normalną kolumną
    out = out.sort_index().reset_index()

    return out

def forecast_cashflow_seasonal(
    monthly: pd.DataFrame,
    periods: int = 6,
    min_history: int = 6,
) -> pd.DataFrame:
    """
    Prognozuje przyszły cashflow na podstawie miesięcznego cashflow.

    Wejście:
        monthly: DataFrame z compute_monthly_cashflow
                 kolumny: ['month', 'income', 'expense', 'net']
        periods: ile kolejnych miesięcy chcemy przewidzieć (default: 6)
        min_history: minimalna liczba miesięcy, żeby użyć modelu "sezonowego";
                     jeśli historii jest mniej, używamy po prostu globalnej średniej.

    Wyjście:
        DataFrame z kolumnami:
            - 'month'   : pierwszy dzień prognozowanego miesiąca
            - 'income'  : NaN (na razie prognozujemy tylko 'net')
            - 'expense' : NaN
            - 'net'     : przewidywane saldo netto
    """
    if len(monthly) < 2:
        raise ValueError("Potrzeba przynajmniej 2 miesięcy historii do prognozy.")

    hist = monthly.copy()
    hist["month"] = pd.to_datetime(hist["month"])
    hist = hist.sort_values("month")

    # Numer miesiąca (1–12) do wyznaczania sezonowości
    hist["month_num"] = hist["month"].dt.month

    # Jeśli historii mamy sporo → średnia po miesiącach roku (styczeń, luty, ...)
    if len(hist) >= min_history:
        seasonal_means = (
            hist.groupby("month_num")["net"]
                .mean()
        )
    else:
        # Jak historii jest mało → jedna globalna średnia
        mean_net = hist["net"].mean()
        seasonal_means = pd.Series(mean_net, index=range(1, 13))

    # Tworzymy kolejne przyszłe miesiące
    last_month = hist["month"].max()
    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),  # miesiąc po ostatnim
        periods=periods,
        freq="MS",  # Month Start
    )

    forecast_rows = []
    for m in future_months:
        m_num = m.month
        # jeśli dla danego miesiąca mamy średnią – bierzemy ją,
        # w przeciwnym razie fallback do globalnej średniej
        if m_num in seasonal_means.index:
            net = seasonal_means.loc[m_num]
        else:
            net = hist["net"].mean()

        forecast_rows.append(
            {
                "month": m,
                "income": pd.NA,   # można rozwinąć w przyszłości, na razie tylko net
                "expense": pd.NA,
                "net": net,
            }
        )

    forecast = pd.DataFrame(forecast_rows)
    return forecast[["month", "income", "expense", "net"]]


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

def prepare_spending_heatmaps(df: pd.DataFrame) -> dict:
    """
    Prepare pivot tables for spending heatmaps.
    Returns dict with:
      - 'dow_month': day-of-week vs month
      - 'dom_month': day-of-month vs month
    """
    if "date" not in df.columns or "amount" not in df.columns:
        return {}

    # Only expenses (amount < 0)
    df_exp = df[df["amount"] < 0].copy()
    if df_exp.empty:
        return {}

    # Positive spending value
    df_exp["spend"] = -df_exp["amount"]

    # Day of week (0=Mon,...,6=Sun)
    df_exp["dow"] = df_exp["date"].dt.dayofweek
    # Day of month
    df_exp["dom"] = df_exp["date"].dt.day
    # Month as YYYY-MM string
    df_exp["month_str"] = df_exp["date"].dt.to_period("M").astype(str)

    # 1) DOW vs month
    pivot_dow_month = df_exp.pivot_table(
        index="dow",
        columns="month_str",
        values="spend",
        aggfunc="sum",
        fill_value=0.0,
    )

    # 2) Day-of-month vs month
    pivot_dom_month = df_exp.pivot_table(
        index="dom",
        columns="month_str",
        values="spend",
        aggfunc="sum",
        fill_value=0.0,
    )

    return {
        "dow_month": pivot_dow_month,
        "dom_month": pivot_dom_month,
    }
