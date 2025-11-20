#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from pathlib import Path

import streamlit as st
import pandas as pd

from finance_analyzer.loader import load_transactions
from finance_analyzer.cleaner import clean_transactions
from finance_analyzer.categorizer import load_category_rules, categorize_transactions
from finance_analyzer.analytics import (
    monthly_income_expense,
    monthly_category_breakdown,
    top_expenses,
)
from finance_analyzer.visuals import (
    plot_monthly_income_expense,
    plot_monthly_expenses_by_category,
    plot_top_expenses,
)


# ---------- HELPER: pipeline ----------

def run_pipeline_from_df(df_raw: pd.DataFrame, rules_path: Path) -> pd.DataFrame:
    """Run cleaning + categorization on a raw dataframe."""
    df_clean = clean_transactions(df_raw)
    rules = load_category_rules(rules_path)
    df_cat = categorize_transactions(df_clean, rules)
    return df_cat


def run_pipeline_from_file(uploaded_file, rules_path: Path) -> pd.DataFrame:
    """
    uploaded_file is a Streamlit UploadedFile or None.
    Uses same loader as CLI for CSV on disk, but for upload we read bytes.
    """
    if uploaded_file is None:
        return None

    # Option 1: if your loader can handle file-like objects, you could adapt it.
    # Here we'll just read with pandas directly and then reuse cleaner + categorizer.

    # IMPORTANT: we assume same encoding as before
    df_raw = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python",
        encoding="cp1250",
    )
    df_cat = run_pipeline_from_df(df_raw, rules_path)
    return df_cat


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Personal Finance Analyzer",
        layout="wide",
    )

    st.title("💸 Personal Finance Analyzer")
    st.markdown(
        "Upload a bank CSV export, categorize transactions, "
        "and explore monthly spending, categories, and top expenses."
    )

    # --- SIDEBAR ---
    st.sidebar.header("Settings")

    rules_default = Path("config/categories_rules.json")
    rules_path_str = st.sidebar.text_input(
        "Category rules JSON path",
        value=str(rules_default),
    )
    rules_path = Path(rules_path_str)

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload bank CSV file",
        type=["csv"],
        help="Use the same bank export format you used in the CLI version.",
    )

    # Filter options (we'll apply them after pipeline)
    year_filter = st.sidebar.text_input("Filter year (optional, e.g. 2025)")
    month_filter = st.sidebar.text_input("Filter month (optional, e.g. 2025-05)")
    top_n = st.sidebar.slider("Top N expenses", min_value=5, max_value=50, value=10, step=5)

    run_button = st.sidebar.button("Run analysis")

    if not run_button:
        st.info("Upload a file and click **Run analysis** to start.")
        return

    if uploaded_file is None:
        st.error("Please upload a CSV file.")
        return

    if not rules_path.exists():
        st.error(f"Category rules file not found: {rules_path}")
        return

    # --- PIPELINE EXECUTION ---
    with st.spinner("Processing file..."):
        try:
            df_cat = run_pipeline_from_file(uploaded_file, rules_path)
        except Exception as e:
            st.error(f"Error while processing file: {e}")
            return

    if df_cat is None or df_cat.empty:
        st.warning("No transactions found after processing.")
        return

    # --- FILTERING ---
    df_view = df_cat.copy()
    if year_filter:
        try:
            y = int(year_filter)
            df_view = df_view[df_view["year"] == y]
        except ValueError:
            st.warning("Year filter ignored (not an integer).")

    if month_filter:
        # expecting format YYYY-MM
        df_view = df_view[df_view["month"].astype(str) == month_filter]

    if df_view.empty:
        st.warning("No transactions after applying filters.")
        return

    # --- MAIN PANELS ---
    st.subheader("Raw transactions (after cleaning & categorization)")
    st.dataframe(
        df_view[["date", "description", "amount", "balance", "currency", "category"]]
        .sort_values("date", ascending=False)
    )

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    total_income = df_view.loc[df_view["type"] == "income", "amount"].sum()
    total_expense = df_view.loc[df_view["type"] == "expense", "amount"].sum()
    net = total_income + total_expense

    col1.metric("Total income", f"{total_income:,.2f}")
    col2.metric("Total expenses", f"{total_expense:,.2f}")
    col3.metric("Net", f"{net:,.2f}")

    # --- ANALYTICS ---
    summary = monthly_income_expense(df_view)
    cat_pivot = monthly_category_breakdown(df_view)
    topn = top_expenses(df_view, n=top_n)

    st.subheader("Monthly income / expenses summary")
    st.dataframe(summary)

    st.subheader("Monthly expenses by category")
    st.dataframe(cat_pivot)

    st.subheader(f"Top {top_n} expenses")
    st.dataframe(topn[["date", "description", "amount", "category"]])

    # --- PLOTS ---
    st.markdown("## 📈 Visualizations")

    # 1) Monthly income / expenses / net
    fig1, _ = plot_monthly_income_expense(summary)
    st.pyplot(fig1)

    # 2) Monthly expenses by category
    fig2, _ = plot_monthly_expenses_by_category(cat_pivot)
    st.pyplot(fig2)

    # 3) Top N expenses
    fig3, _ = plot_top_expenses(topn)
    st.pyplot(fig3)

    # Optional: download cleaned & categorized CSV
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned & categorized CSV",
        data=csv_bytes,
        file_name="transactions_clean_categorized.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

