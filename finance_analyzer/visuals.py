# finance_analyzer/visuals.py
import pandas as pd
import matplotlib.pyplot as plt

def _prepare_month_labels(month_series: pd.Series) -> list[str]:
    """
    Convert Period[M] or datetime month column into readable labels: '2025-05', '2025-06', ...
    """
    if hasattr(month_series.dtype, "freq"):  # PeriodDtype
        return month_series.astype(str).tolist()
    else:
        # Assume datetime-like
        return month_series.dt.to_period("M").astype(str).tolist()


def plot_monthly_income_expense(summary_df: pd.DataFrame):
    """
    summary_df: output of monthly_income_expense(df)
      must have columns: ['month', 'income', 'expense', 'net']
    """
    labels = _prepare_month_labels(summary_df["month"])
    income = summary_df.get("income", pd.Series(0, index=summary_df.index))
    expense = summary_df.get("expense", pd.Series(0, index=summary_df.index))
    net = summary_df.get("net", pd.Series(0, index=summary_df.index))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bar for income and expense
    x = range(len(labels))
    ax.bar(x, income, label="Income")
    ax.bar(x, expense, label="Expenses")  # expenses are negative

    # Net as line
    ax.plot(x, net, marker="o", label="Net")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Amount")
    ax.set_title("Monthly Income / Expenses / Net")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_monthly_expenses_by_category(pivot: pd.DataFrame):
    """
    pivot: output of monthly_category_breakdown(df)
      index: month, columns: category, values: negative expenses
    """
    # Convert index to labels
    labels = _prepare_month_labels(pivot.index.to_series())

    # Make values positive for plotting (expenses are negative)
    data = -pivot  # minus sign

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(labels))
    bottom = [0] * len(labels)

    for category in data.columns:
        values = data[category].tolist()
        ax.bar(x, values, bottom=bottom, label=category)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Expenses (positive)")
    ax.set_title("Monthly Expenses by Category (stacked)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_top_expenses(top_df: pd.DataFrame):
    """
    top_df: output of top_expenses(df, n)
      must have ['description', 'amount']
    """
    # Sort by amount (most negative at bottom)
    top_df = top_df.sort_values("amount")

    descriptions = top_df["description"].tolist()
    amounts = top_df["amount"].tolist()

    # Make them positive for plotting
    values = [-a for a in amounts]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(descriptions))

    ax.barh(y, values)
    ax.set_yticks(list(y))
    ax.set_yticklabels(descriptions)
    ax.invert_yaxis()  # biggest at top

    ax.set_xlabel("Expense amount (positive)")
    ax.set_title("Top Expenses")
    fig.tight_layout()
    return fig, ax
