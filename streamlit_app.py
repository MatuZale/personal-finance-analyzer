#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from finance_analyzer.loader import load_transactions
from finance_analyzer.cleaner import clean_transactions
from finance_analyzer.analytics import (
    monthly_income_expense,
    monthly_category_breakdown,
    top_expenses,
    compute_monthly_cashflow,
    forecast_cashflow_seasonal,
)
from finance_analyzer.visuals import (
    plot_monthly_income_expense,
    plot_monthly_expenses_by_category,
    plot_top_expenses,
)
from finance_analyzer.recurring import find_recurring_transactions, _normalize_description

from finance_analyzer.categorizer import (
    load_category_rules,
    categorize_transactions,
    load_manual_overrides,
    apply_manual_overrides,
)

import warnings

# Suppress specific pandas warnings in Streamlit app
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="pandas.core.reshape.concat",
)


# ---------- HELPER: pipeline ----------

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust reader for Streamlit UploadedFile.
    Tries multiple encodings commonly used in PL bank exports.
    """
    raw = uploaded_file.read()  # bytes

    # Try several encodings
    encodings_to_try = ["utf-8-sig", "cp1250", "iso-8859-2", "latin1"]

    last_error = None
    for enc in encodings_to_try:
        try:
            buffer = io.StringIO(raw.decode(enc))
            df = pd.read_csv(buffer, sep=None, engine="python")
            st.info(f"Successfully loaded with encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not decode CSV with tried encodings: {encodings_to_try}. Last error: {last_error}")


def run_pipeline_from_df(
    df_raw: pd.DataFrame,
    rules_path: Path,
    overrides_path: Path,
) -> pd.DataFrame:
    """Run cleaning + categorization (+ manual overrides) on a raw dataframe."""
    df_clean = clean_transactions(df_raw)
    rules = load_category_rules(rules_path)
    overrides = load_manual_overrides(overrides_path)

    df_cat = categorize_transactions(df_clean, rules)
    df_cat = apply_manual_overrides(df_cat, overrides)
    return df_cat


def run_pipeline_from_file(
    uploaded_file,
    rules_path: Path,
    overrides_path: Path,
) -> pd.DataFrame:
    """
    uploaded_file is a Streamlit UploadedFile or None.
    """
    if uploaded_file is None:
        return None

    df_raw = read_uploaded_csv(uploaded_file)
    df_cat = run_pipeline_from_df(df_raw, rules_path, overrides_path)
    return df_cat

def show_cashflow_forecast_tab(df: pd.DataFrame):
    st.header("Forcast of monthly cashflow")

    # 1. History
    monthly = compute_monthly_cashflow(df)

    # 2. Parameters from user
    col1, col2 = st.columns(2)
    with col1:
        periods = st.slider("Number of months to forcast", 3, 24, 6, step=1)
    with col2:
        min_history = st.slider("Minimal number of months", 3, 24, 6, step=1)

    # 3. Prognose
    forecast = forecast_cashflow_seasonal(
        monthly,
        periods=periods,
        min_history=min_history,
    )

    hist_plot = monthly.assign(kind="history")
    fc_plot = forecast.assign(kind="forecast")

    frames = [hist_plot, fc_plot]

# Remove empty frames
    clean_frames = []
    for f in frames:
        if f is None:
            continue
        if f.empty:
            continue
    # if all columns are NA
        if f.dropna(how="all").empty:
            continue
        clean_frames.append(f)

    if clean_frames:
        combined = pd.concat(clean_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["month", "income", "expense", "net", "kind"])

    # 4. plot - net cashflow
    fig, ax = plt.subplots(figsize=(10, 4))

    # history
    ax.bar(
        hist_plot["month"],
        hist_plot["net"],
        width=20,
        label="History",
    )

    # history (half-transparent)
    ax.bar(
        fc_plot["month"],
        fc_plot["net"],
        width=20,
        alpha=0.6,
        label="Forecast",
        linestyle="--",
    )

    ax.set_title("Monthly net cashflow (history + forecast)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net amount")

    ax.legend()

    # dark-mode
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig)

    st.subheader("Monthly data (historical + forecast)")
    st.dataframe(
    combined[["month", "income", "expense", "net", "kind"]]
    .sort_values("month")
    .reset_index(drop=True)
    )

# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Personal Finance Analyzer",
        layout="wide",
    )

    st.title("Personal Finance Analyzer")
    st.markdown(
        "Upload a bank CSV export, categorize transactions, "
        "and explore monthly spending, categories, and top expenses."
    )

    # --- SIDEBAR ---
    st.sidebar.header("Settings")

    # Paths for rules + manual overrides
    rules_default = Path("config/categories_rules.json")
    rules_path_str = st.sidebar.text_input(
        "Category rules JSON path",
        value=str(rules_default),
    )
    rules_path = Path(rules_path_str)

    overrides_default = Path("config/manual_overrides.csv")
    overrides_path_str = st.sidebar.text_input(
        "Manual overrides CSV path",
        value=str(overrides_default),
    )
    manual_overrides_path = Path(overrides_path_str)

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

    # --- PIPELINE EXECUTION (tylko gdy klikniesz przycisk) ---
    if run_button:
        if uploaded_file is None:
            st.error("Please upload a CSV file.")
            return

        if not rules_path.exists():
            st.error(f"Category rules file not found: {rules_path}")
            return

        with st.spinner("Processing file..."):
            try:
                df_cat = run_pipeline_from_file(
                    uploaded_file, rules_path, manual_overrides_path
                )
            except Exception as e:
                st.error(f"Error while processing file: {e}")
                return

        if df_cat is None or df_cat.empty:
            st.warning("No transactions found after processing.")
            return

        # ZAPISZ wynik w session_state
        st.session_state["df_cat"] = df_cat

    # --- JEŚLI JESZCZE NIC NIE POLICZYLIŚMY ---
    if "df_cat" not in st.session_state:
        st.info("Upload a file and click **Run analysis** to start.")
        return

    # ODTĄD ZAWSZE KORZYSTAMY Z ZAPISANEGO DF
    df_cat = st.session_state["df_cat"]


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

    base_cols = ["date", "description", "amount", "balance", "currency", "category"]

    # If merchant exists, insert it after date
    display_cols = ["date"]
    if "merchant" in df_view.columns:
        display_cols.append("merchant")
    display_cols += [c for c in base_cols if c not in ("date", "merchant")]

    st.dataframe(
        df_view[display_cols].sort_values("date", ascending=False)
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

        # --- ANALYTICS ---
    summary = monthly_income_expense(df_view)
    cat_pivot = monthly_category_breakdown(df_view)
    topn = top_expenses(df_view, n=top_n)
    recurring = find_recurring_transactions(df_view)

    # Przygotuj dane pojedynczych płatności do podglądu cykli
    expenses_with_norm = df_view[df_view["amount"] < 0].copy()
    if not expenses_with_norm.empty:
        expenses_with_norm["desc_norm"] = _normalize_description(
            expenses_with_norm["description"]
        )

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary", "Category", "Plots", "Forecast cashflow"]
    )

    with tab1:
        st.subheader("Monthly income / expenses summary")
        st.dataframe(summary)

        st.subheader(f"Top {top_n} expenses")
        st.dataframe(topn[["date", "description", "amount", "category"]])

    with tab2:
        st.subheader("Monthly expenses by category")
        st.dataframe(cat_pivot)

    with tab3:
        st.markdown("Visualizations")
        st.subheader("Wykresy")

        if df_view.empty:
            st.info("Brak danych do wykresów – zmień filtry u góry.")
        else:
            # 1) Monthly income / expense
            fig1, _ = plot_monthly_income_expense(summary)
            st.pyplot(fig1)

            # 2) Monthly expenses by category
            fig2, _ = plot_monthly_expenses_by_category(cat_pivot)
            st.pyplot(fig2)

            # 3) Top N expenses
            fig3, _ = plot_top_expenses(topn)
            st.pyplot(fig3)

    with tab4:
        # NOWA ZAKŁADKA — prognoza cashflow
        show_cashflow_forecast_tab(df_view)

    # --- RECURRING PAYMENTS ---
    st.subheader("Recurring payments (subscriptions, rent, etc.)")

    if recurring.empty:
        st.info(
            "No clear recurring patterns detected with current settings. "
            "Try using a longer date range or adjust the detection heuristics."
        )
    else:
        # Table summary
        st.markdown("### Summary of detected recurring payments")
        st.dataframe(
            recurring[
                [
                    "example_description",
                    "category",
                    "n_payments",
                    "avg_amount",
                    "mean_interval_days",
                    "first_date",
                    "last_date",
                ]
            ]
        )

        # Details per series
        st.markdown("### Detailed transactions for each recurring series")

        if expenses_with_norm.empty:
            st.info("No expense transactions available to show details.")
        else:
            for _, row in recurring.iterrows():
                desc_norm = row["desc_norm"]

                # All transactions in this series
                series_tx = (
                    expenses_with_norm[expenses_with_norm["desc_norm"] == desc_norm]
                    .sort_values("date")
                    .reset_index(drop=True)
                )

                # Header with example description + n payments + avg amount
                title = (
                    f"{row['example_description']} "
                    f"({row['n_payments']}×, "
                    f"avg {row['avg_amount']:.2f})"
                )

                with st.expander(title, expanded=False):
                    st.dataframe(
                        series_tx[
                            [
                                "date",
                                "description",
                                "amount",
                                "category",
                            ]
                        ]
                    )


    # ===================== CATEGORY EDITOR ===================== #
    st.markdown("## 🧩 Interactive category editor (manual overrides)")

    with st.expander("Edit categories for repeating descriptions", expanded=False):
        # Load current overrides into session state (so we can modify them)
        if "overrides_df" not in st.session_state:
            st.session_state["overrides_df"] = load_manual_overrides(manual_overrides_path)

        overrides_df = st.session_state["overrides_df"]

        st.markdown("### Current manual overrides")
        if overrides_df.empty:
            st.write("No manual overrides defined yet.")
        else:
            st.dataframe(overrides_df)

        # Focus on 'Uncategorized' descriptions in the current filtered view
        uncat = df_view[df_view["category"] == "Uncategorized"].copy()
        # Build suggestions based on already categorized transactions
        known = df_view[df_view["category"] != "Uncategorized"].copy()
        merchant_suggestions = {}
        if not known.empty and "merchant" in known.columns:
            merchant_suggestions = (
                known.groupby("merchant")["category"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )

        if uncat.empty:
            st.success("No 'Uncategorized' transactions in the current view.")
        else:
            st.markdown("### Add or update an override")

            grouped = (
                uncat.groupby("description")
                    .agg(
                        n=("amount", "size"),
                        total_amount=("amount", "sum"),
                    )
                    .reset_index()
                    .sort_values("n", ascending=False)
            )

            st.caption(
                "Pick a description that appears often and assign it a category. "
                "The rule will match this text as a substring in future runs."
            )

            desc_options = grouped["description"].tolist()
            chosen_desc = st.selectbox(
                "Choose a description to override",
                options=desc_options,
                key="override_desc_select",
            )

            # Find merchant for this description (use most common merchant if multiple)
            chosen_merchant = None
            if "merchant" in uncat.columns:
                candidates = uncat.loc[
                    uncat["description"] == chosen_desc, "merchant"
                ]
                if not candidates.empty:
                    chosen_merchant = candidates.mode().iloc[0]

            # Suggested category based on previous data for this merchant
            suggested_cat = None
            if chosen_merchant and chosen_merchant in merchant_suggestions:
                suggested_cat = merchant_suggestions[chosen_merchant]
                st.caption(
                    f"Suggested category for merchant **{chosen_merchant}**: "
                    f"**{suggested_cat}**"
                )

            # Available categories from current data (except Uncategorized)
            existing_cats = sorted(
                [c for c in df_view["category"].dropna().unique() if c != "Uncategorized"]
            )

            # Default dropdown index
            default_index = 0  # "(none)"
            if suggested_cat and suggested_cat in existing_cats:
                default_index = 1 + existing_cats.index(suggested_cat)

            chosen_existing = st.selectbox(
                "Assign existing category",
                options=["(none)"] + existing_cats,
                index=default_index,
                key="override_existing_cat",
            )

            new_cat = st.text_input("Or type a new category", value="")

            if chosen_existing != "(none)":
                final_cat = chosen_existing
            elif new_cat.strip():
                final_cat = new_cat.strip()
            else:
                final_cat = None

            if st.button("Add / update override"):
                if final_cat is None:
                    st.warning("Please choose an existing category or type a new one.")
                else:
                    substr = chosen_desc  # we use full description as substring pattern

                    if overrides_df.empty:
                        overrides_df = pd.DataFrame(
                            [{"substring": substr, "category": final_cat}]
                        )
                    else:
                        mask = overrides_df["substring"] == substr
                        if mask.any():
                            overrides_df.loc[mask, "category"] = final_cat
                        else:
                            overrides_df = pd.concat(
                                [
                                    overrides_df,
                                    pd.DataFrame(
                                        [{"substring": substr, "category": final_cat}]
                                    ),
                                ],
                                ignore_index=True,
                            )

                    # Save to CSV
                    try:
                        overrides_df.to_csv(manual_overrides_path, index=False)
                        st.session_state["overrides_df"] = overrides_df
                        st.success(
                            f"Saved override: '{substr}' → '{final_cat}'. "
                            "Re-run analysis to apply it."
                        )
                    except Exception as e:
                        st.error(f"Could not save overrides: {e}")

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
