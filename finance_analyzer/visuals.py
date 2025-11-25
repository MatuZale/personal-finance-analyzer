# finance_analyzer/visuals.py

import matplotlib
matplotlib.use("Agg")  # safe for Streamlit / headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


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
