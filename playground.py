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
import matplotlib.pyplot as plt

csv_path = "data/raw/your_file.csv"
rules_path = "config/categories_rules.json"

# --- pipeline ---
df_raw = load_transactions(csv_path)
df_clean = clean_transactions(df_raw)
rules = load_category_rules(rules_path)
df_cat = categorize_transactions(df_clean, rules)

# --- analytics ---
summary = monthly_income_expense(df_cat)
cat_pivot = monthly_category_breakdown(df_cat)
top10 = top_expenses(df_cat, n=10)

print("\n--- MONTHLY SUMMARY ---")
print(summary)

print("\n--- CATEGORY PIVOT (first 5 rows) ---")
print(cat_pivot.head())

print("\n--- TOP 10 EXPENSES ---")
print(top10[["date", "description", "amount", "category"]])

# --- plots ---
fig1, ax1 = plot_monthly_income_expense(summary)
fig1.savefig("reports/monthly_income_expense.png", dpi=200)

fig2, ax2 = plot_monthly_expenses_by_category(cat_pivot)
fig2.savefig("reports/monthly_expenses_by_category.png", dpi=200)

fig3, ax3 = plot_top_expenses(top10)
fig3.savefig("reports/top_expenses.png", dpi=200)

# Optional: show in an interactive window if running locally
# plt.show()
