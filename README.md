💸 Personal Finance Analyzer

A Python tool for analyzing personal finance using bank CSV exports

This project processes bank transaction exports (CSV files), cleans them, categorizes expenses automatically, generates analytics, produces visual visualizations, and offers a Streamlit web interface for interactive exploration.

It works with Polish bank exports (including CP1250/ISO-8859-2 encodings), but is fully extendable to other formats.

🚀 Features
✔ 1. Robust CSV ingestion

Automatically detects CSV delimiters (, or ;).
Handles Polish encodings (cp1250, iso-8859-2).
Parses transaction dates and Polish-style numbers (1 234,56 zł).

✔ 2. Transaction cleaning and standardization

Standard schema with columns:
date, description, amount, balance, currency, type (income/expense), year, month
Automatically merges counterparty / title / details into a single, clean description.
Removes empty/unnamed columns.

✔ 3. Automatic transaction categorization

Rule-based keyword matching (editable JSON file).
Categories include: Groceries, Transport, Shopping, Salary, Rent, Utilities, Restaurants, etc.
Supports manual overrides.

✔ 4. Analytics

Monthly income vs expenses
Monthly expenses by category (stacked bars)
Top N expenses
Net flow and spending patterns

✔ 5. Beautiful visualizations

Monthly income/expenses bar + net line plot
Category-based stacked bar
Top expenses horizontal bar chart
All plots saved automatically as PNGs

✔ 6. Command-Line Interface (CLI)

Run the full pipeline with:

python -m finance_analyzer analyze data/raw/my_bank_export.csv \
    --rules config/categories_rules.json \
    --out reports/ \
    --top 15


Generates PNG plots and printable summaries.

✔ 7. Streamlit Web App

Interactive interface:

python -m streamlit run streamlit_app.py


Features:

Upload CSV
Apply year/month filters
View categorized transactions
Metrics (income, expenses, net)
Embedded plots
Download cleaned & categorized CSV

📂 Project Structure
fin_cont/
├── finance_analyzer/
│   ├── loader.py             # CSV loader with encoding detection
│   ├── cleaner.py            # Cleaning & normalization
│   ├── categorizer.py        # Rule-based categorization
│   ├── analytics.py          # Summaries & insights
│   ├── visuals.py            # Visualization functions (matplotlib)
│   ├── __main__.py           # CLI entry point
│
├── config/
│   └── categories_rules.json # Category rules
│
├── data/
│   └── raw/                  # Bank CSV exports (ignored in Git)
│
├── reports/                  # Saved plots & outputs
│
├── streamlit_app.py          # Streamlit UI
├── playground.py             # Development sandbox
└── README.md                 # You're reading this

🔧 Installation
1. Create virtual environment (recommended)
conda create -n fin_cont_env python=3.11
conda activate fin_cont_env

2. Install dependencies
python -m pip install streamlit pandas matplotlib numba pyarrow


(Or create requirements.txt later.)

▶️ Usage
CLI mode
python -m finance_analyzer analyze path/to/export.csv \
    --rules config/categories_rules.json \
    --out reports \
    --top 15

Streamlit UI
python -m streamlit run streamlit_app.py


Then open:
👉 http://localhost:8501

🧠 How categorization works

Category rules are stored in:

config/categories_rules.json

Example:

{
  "Groceries": ["biedronka", "lidl", "zabka", "kaufland"],
  "Transport": ["uber", "bolt", "pkp"],
  "Shopping":  ["allegro", "amazon"],
  "Salary":    ["wynagrodzenie", "salary"]
}


You can edit this file anytime to improve categorization accuracy.

📊 Example Visualizations


Income vs expenses
Category expenses per month
Biggest expenses

🧩 TODO / Future improvements (optional section)

Recurring payment detection (subscriptions)
Forecast monthly financial behavior
Machine-learning-based categorization
Export full PDF financial reports

📜 License

MIT

✍️ Author

Tomasz Młynik