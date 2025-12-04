# finance_analyzer/visuals.py

import matplotlib
matplotlib.use("Agg")  # safe for Streamlit / headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ---------- COMMON DARK STYLE HELPERS ----------

BG_COLOR = "#05060a"
AX_BG_COLOR = "#0b0f1a"
GRID_COLOR = "#2a3348"
TEXT_COLOR = "#e5eaf5"
ACCENT_POS = "#3dd2c8"   # income, positive stuff
ACCENT_NEG = "#ff6b81"   # expenses, losses
ACCENT_NET = "#ffd166"   # net line
ACCENT_MISC = "#7f5af0"  # extra


def _money_formatter(x, pos):
    """Format y-axis as e.g. 1 234,56"""
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")


def _setup_dark_ax(figsize=(8, 4)):
    """Create a dark-themed figure + axes with consistent styling."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)

    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    ax.yaxis.set_major_formatter(FuncFormatter(_money_formatter))
    return fig, ax


# ---------- PLOT 1: MONTHLY INCOME / EXPENSE / NET ----------

def plot_monthly_income_expense(summary_df):
    """
    Expects a DataFrame with columns:
      - 'month' (string or Period, like '2025-01')
      - 'income'
      - 'expense' (negative values or absolute, we handle both)
      - 'net'
    """
    if summary_df is None or summary_df.empty:
        return None, None

    df = summary_df.copy()
    # ensure month is sorted nicely
    if "month" in df.columns:
        df = df.sort_values("month")

    months = df["month"].astype(str).tolist()
    income = df["income"].values
    expense = df["expense"].values
    net = df["net"].values

    # If expenses are negative, flip them for plotting
    if np.any(expense < 0):
        expense_plot = -expense
    else:
        expense_plot = expense

    x = np.arange(len(months))

    fig, ax = _setup_dark_ax(figsize=(9, 4))

    width = 0.35
    ax.bar(
        x - width / 2,
        income,
        width=width,
        color=ACCENT_POS,
        alpha=0.9,
        label="Income",
    )
    ax.bar(
        x + width / 2,
        expense_plot,
        width=width,
        color=ACCENT_NEG,
        alpha=0.9,
        label="Expenses",
    )

    ax.plot(
        x,
        net,
        color=ACCENT_NET,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Net",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", color=TEXT_COLOR)
    ax.set_title("Monthly income, expenses and net", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Amount", color=TEXT_COLOR)

    ax.legend(facecolor=AX_BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    return fig, ax


# ---------- PLOT 2: MONTHLY EXPENSES BY CATEGORY ----------

def plot_monthly_expenses_by_category(cat_pivot):
    """
    Expects a DataFrame indexed by 'month' or with a 'month' column,
    columns are categories, values are *expenses* (usually positive absolute values).
    """
    if cat_pivot is None or cat_pivot.empty:
        return None, None

    df = cat_pivot.copy()

    # If month is a column, move it to index
    if "month" in df.columns:
        df = df.set_index("month")

    df = df.sort_index()
    months = df.index.astype(str).tolist()
    x = np.arange(len(months))

    categories = df.columns.tolist()

    fig, ax = _setup_dark_ax(figsize=(9, 4))

    # build a simple palette — variations between accent colors
    base_colors = [ACCENT_NEG, ACCENT_MISC, ACCENT_POS, ACCENT_NET]
    colors = []
    for i, cat in enumerate(categories):
        colors.append(base_colors[i % len(base_colors)])

    bottom = np.zeros(len(df))

    for i, cat in enumerate(categories):
        values = df[cat].fillna(0).values
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[i],
            alpha=0.85,
            label=cat,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", color=TEXT_COLOR)
    ax.set_title("Monthly expenses by category", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Amount", color=TEXT_COLOR)

    ax.legend(
        ncol=2,
        fontsize=8,
        facecolor=AX_BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    fig.tight_layout()
    return fig, ax


# ---------- PLOT 3: TOP N EXPENSES ----------

def plot_top_expenses(topn_df):
    """
    Expects a DataFrame with at least:
      - 'description'
      - 'amount' (negative or positive; we plot magnitude)
      - optional 'category', 'date'
    """
    if topn_df is None or topn_df.empty:
        return None, None

    df = topn_df.copy()

    # Take magnitude of expenses
    amounts = df["amount"].values
    if np.any(amounts < 0):
        values = -amounts
    else:
        values = amounts

    # Shorten description for plotting
    labels = []
    for desc in df["description"].astype(str).tolist():
        if len(desc) > 40:
            labels.append(desc[:37] + "...")
        else:
            labels.append(desc)

    # Reverse for nicer horizontal bar order (largest on top)
    idx = np.arange(len(df))[::-1]
    values = values[::-1]
    labels = labels[::-1]

    fig, ax = _setup_dark_ax(figsize=(9, 5))

    bars = ax.barh(idx, values, color=ACCENT_NEG, alpha=0.9)

    ax.set_yticks(idx)
    ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Amount", color=TEXT_COLOR)
    ax.set_title("Top expenses", color=TEXT_COLOR, fontsize=11)

    # optional: annotate values at the end of bars
    for rect, val in zip(bars, values):
        width = rect.get_width()
        ax.text(
            width * 1.01,
            rect.get_y() + rect.get_height() / 2,
            _money_formatter(val, None),
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=7,
        )

    fig.tight_layout()
    return fig, ax


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def detailed_time_series_analysis(df: pd.DataFrame) -> None:
    """
    Interaktywna analiza czasowa:
    - wybór metryki (saldo, netto, income, expense, saldo skumulowane)
    - wybór zakresu dat (ROI)
    - statystyki dopasowane do typu metryki
    - regresja wielomianowa (1–6 stopień) i trend na wykresie
    """
    st.subheader("📈 Detailed time series analysis")

    df = df.copy()
    df = df.sort_values("date")

    if "date" not in df.columns:
        st.error("Brak kolumny 'date' w DataFrame.")
        return

    # --------- PRZYGOTOWANIE DANYCH BAZOWYCH ---------
    # Saldo dzienne – ostatnie saldo danego dnia
    daily_balance = None
    if "balance" in df.columns:
        daily_balance = (
            df.sort_values("date")
              .groupby("date", as_index=False)["balance"]
              .last()
        )

    # Dzienny przepływ netto + income + expense
    daily_net = (
        df.groupby("date", as_index=False)["amount"]
          .sum()
          .rename(columns={"amount": "net"})
    )

    # Income (tylko dodatnie) i expense (tylko ujemne)
    if "type" in df.columns:
        df_income = (
            df[df["type"] == "income"]
            .groupby("date", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "income"})
        )
        df_expense = (
            df[df["type"] == "expense"]
            .groupby("date", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "expense"})
        )
    else:
        # fallback gdy brak kolumny "type"
        df_income = (
            df[df["amount"] > 0]
            .groupby("date", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "income"})
        )
        df_expense = (
            df[df["amount"] < 0]
            .groupby("date", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "expense"})
        )

    # Sklej wszystko w jedną ramkę dzienną
    daily = daily_net.set_index("date")
    daily = daily.join(df_income.set_index("date"), how="left")
    daily = daily.join(df_expense.set_index("date"), how="left")
    daily = daily.fillna(0.0)
    daily = daily.reset_index()

    # Dodatkowo: dzienny wydatek jako wartość dodatnia
    daily["expense_abs"] = -daily["expense"]  # bo expense jest ujemny

    # Saldo skumulowane (cumsum przepływu netto)
    daily["cum_net"] = daily["net"].cumsum()

    # --------- WYBÓR METRYKI ---------
    metric_options = [
        "Saldo po transakcji (balance)",
        "Dzienny przepływ netto (suma amount z dnia)",
        "Dzienny dochód (tylko income)",
        "Dzienny wydatek (tylko expense, wartości dodatnie)",
        "Saldo skumulowane (cumsum przepływu netto)",
    ]

    metric = st.selectbox(
        "Wybierz co analizujemy:",
        metric_options,
    )

    # Dobierz serię do analizy
    if metric == "Saldo po transakcji (balance)":
        if daily_balance is None:
            st.error("Brak kolumny 'balance' dla tej metryki.")
            return
        series = daily_balance.rename(columns={"balance": "value"})
        y_label = "Saldo"
        is_level_series = True  # poziom w czasie (saldo)
    elif metric == "Dzienny przepływ netto (suma amount z dnia)":
        series = daily[["date", "net"]].rename(columns={"net": "value"})
        y_label = "Dzienny przepływ netto"
        is_level_series = False  # to jest przepływ
    elif metric == "Dzienny dochód (tylko income)":
        series = daily[["date", "income"]].rename(columns={"income": "value"})
        y_label = "Dzienny dochód"
        is_level_series = False
    elif metric == "Dzienny wydatek (tylko expense, wartości dodatnie)":
        series = daily[["date", "expense_abs"]].rename(columns={"expense_abs": "value"})
        y_label = "Dzienny wydatek (>= 0)"
        is_level_series = False
    else:  # "Saldo skumulowane (cumsum przepływu netto)"
        series = daily[["date", "cum_net"]].rename(columns={"cum_net": "value"})
        y_label = "Saldo skumulowane"
        is_level_series = True

    if series.empty:
        st.warning("Brak danych do analizy.")
        return

    # --------- REGION OF INTEREST (ROI) ---------
    date_min = series["date"].min()
    date_max = series["date"].max()

    st.markdown("### 🕒 Region of interest (ROI)")
    start_date, end_date = st.slider(
    "Wybierz zakres dat:",
    min_value=date_min.to_pydatetime(),
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="DD/MM/YYYY",
    key="roi_date_range_slider",
    )

    mask = (series["date"] >= pd.to_datetime(start_date)) & (
        series["date"] <= pd.to_datetime(end_date)
    )
    roi = series.loc[mask].copy()

    if roi.empty:
        st.warning("Wybrany zakres dat nie zawiera danych.")
        return

    # --------- STATYSTYKI W ROI ---------
    st.markdown("### 📊 Podstawowe statystyki w ROI")

    if is_level_series:
        # dla serii poziomów (saldo itp.) ma sens: początek, koniec, zmiana, min, max
        start_val = roi["value"].iloc[0]
        end_val = roi["value"].iloc[-1]
        delta = end_val - start_val
        vmin = roi["value"].min()
        vmax = roi["value"].max()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Saldo na początku", f"{start_val:,.2f}")
        with col2:
            st.metric("Saldo na końcu", f"{end_val:,.2f}")
        with col3:
            st.metric("Zmiana salda", f"{delta:,.2f}")
        with col4:
            st.metric("Min", f"{vmin:,.2f}")
        with col5:
            st.metric("Max", f"{vmax:,.2f}")
    else:
        # dla serii przepływów: suma, średnia, min, max
        total = roi["value"].sum()
        mean = roi["value"].mean()
        vmin = roi["value"].min()
        vmax = roi["value"].max()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Suma (netto w ROI)", f"{total:,.2f}")
        with col2:
            st.metric("Średnia dziennie", f"{mean:,.2f}")
        with col3:
            st.metric("Min", f"{vmin:,.2f}")
        with col4:
            st.metric("Max", f"{vmax:,.2f}")

    # --------- REGRESJA WIELOMIANOWA (1–6) ---------
    st.markdown("### 📐 Trend (regresja wielomianowa)")

    # pozwól na większy stopień, ale max 6 (i tak trzeba uważać na overfitting)
    degree = st.slider(
    "Stopień wielomianu",
    min_value=1,
    max_value=6,
    value=2,
    help="Uwaga: wysokie stopnie łatwo się przeuczają – patrz na wykres krytycznie.",
    key="poly_degree_slider",
    )


    trend_df = None
    if len(roi) >= degree + 1:
        # Oś X jako dni od początku ROI (żeby liczby były małe)
        x = (roi["date"] - roi["date"].min()).dt.total_seconds() / 86400.0
        y = roi["value"].values

        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)

        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = poly(x_fit)

        trend_df = pd.DataFrame(
            {
                "date": roi["date"].min() + pd.to_timedelta(x_fit, unit="D"),
                "value": y_fit,
            }
        )

        # Info o kierunku trendu (tylko dla 1 stopnia)
        if degree == 1:
            slope_per_day = coeffs[0]
            st.info(f"Slope (kierunek trendu) ≈ {slope_per_day:.4f} jednostek / dzień")
    else:
        st.warning(
            "Za mało punktów w ROI, aby dopasować wielomian tego stopnia."
        )

    # --------- WYKRES (Plotly, dark mode) ---------
    st.markdown("### 🔍 Wykres z wyróżnionym ROI i trendem")

    fig = go.Figure()

    # Cała seria (lekko przygaszona)
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["value"],
            mode="lines",
            name="Cały zakres",
            opacity=0.3,
        )
    )

    # ROI
    fig.add_trace(
        go.Scatter(
            x=roi["date"],
            y=roi["value"],
            mode="lines+markers",
            name="ROI",
        )
    )

    # Trend
    if trend_df is not None:
        fig.add_trace(
            go.Scatter(
                x=trend_df["date"],
                y=trend_df["value"],
                mode="lines",
                name=f"Trend (poly deg={degree})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Data",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")

    # --------- Histogram dla metryk przepływów ---------
    if not is_level_series:
        with st.expander("📦 Histogram wartości w ROI"):
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=roi["value"], nbinsx=30, name="ROI values"))
            hist_fig.update_layout(
                template="plotly_dark",
                xaxis_title=y_label,
                yaxis_title="Liczba dni",
                bargap=0.05,
            )
            st.plotly_chart(hist_fig, width="stretch")