# finance_analyzer/__main__.py

import argparse
from pathlib import Path

from .loader import load_transactions
from .cleaner import clean_transactions
from .categorizer import load_category_rules, categorize_transactions
from .analytics import (
    monthly_income_expense,
    monthly_category_breakdown,
    top_expenses,
)
from .visuals import (
    plot_monthly_income_expense,
    plot_monthly_expenses_by_category,
    plot_top_expenses,
)


def cmd_analyze(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    rules_path = Path(args.rules)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {csv_path}")
    df_raw = load_transactions(csv_path) # type: ignore

    print("[INFO] Cleaning...")
    df_clean = clean_transactions(df_raw)

    print(f"[INFO] Loading category rules from: {rules_path}")
    rules = load_category_rules(rules_path)

    print("[INFO] Categorizing...")
    df_cat = categorize_transactions(df_clean, rules)

    # --- Analytics ---
    print("[INFO] Computing analytics...")
    summary = monthly_income_expense(df_cat)
    cat_pivot = monthly_category_breakdown(df_cat)
    topn = top_expenses(df_cat, n=args.top)

    print("\n=== Monthly income/expenses summary ===")
    print(summary)

    print("\n=== Category breakdown (first 5 rows) ===")
    print(cat_pivot.head())

    print(f"\n=== Top {args.top} expenses ===")
    print(topn[["date", "description", "amount", "category"]])

    # --- Plots ---
    print("[INFO] Generating plots...")
    fig1, _ = plot_monthly_income_expense(summary)
    fig1.savefig(out_dir / "monthly_income_expense.png", dpi=200)

    fig2, _ = plot_monthly_expenses_by_category(cat_pivot)
    fig2.savefig(out_dir / "monthly_expenses_by_category.png", dpi=200)

    fig3, _ = plot_top_expenses(topn)
    fig3.savefig(out_dir / "top_expenses.png", dpi=200)

    print(f"[INFO] Plots saved in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="finance_analyzer",
        description="Personal finance analyzer from bank CSV exports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze subcommand ---
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a CSV file and generate reports.",
    )
    p_analyze.add_argument(
        "csv",
        help="Path to bank CSV export.",
    )
    p_analyze.add_argument(
        "--rules",
        default="config/categories_rules.json",
        help="Path to JSON file with category rules.",
    )
    p_analyze.add_argument(
        "--out",
        default="reports",
        help="Output directory for plots.",
    )
    p_analyze.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top expenses to show.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
