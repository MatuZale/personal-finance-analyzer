📘 Personal Finance Analyzer (Python + Streamlit)

A complete personal finance analysis toolkit built in Python.
It imports bank CSV exports, cleans them, categorizes transactions, computes analytics, generates charts, and provides an interactive Streamlit web app.

Perfect for data analysis practice, automation, and portfolio demonstration.

🚀 Features
✔ Robust CSV loader

Auto-detects delimiters (, or ;)

Handles Polish encodings (cp1250, iso-8859-2)

Parses Polish-style numeric formats (1 234,56 zł)

✔ Smart transaction cleaning

Transforms messy bank exports into a clean, standard schema:

date, description, amount, balance, currency

type (income/expense)

year, month

Merges fields like counterparty, title, and details into a single description.

✔ Automatic categorization

Based on configurable categories_rules.json:

Groceries, Transport, Restaurants, Shopping

Salary, Rent, Utilities

Supports manual overrides

Easy to extend

✔ Analytics

Monthly income / expense summary

Monthly category breakdown

Top N expenses

Net flow over time

✔ Visualizations

Monthly income vs expense bar + net line

Stacked bar chart of category expenses

Horizontal bar chart for top expenses

All saved as PNG files.

✔ Command-Line Interface (CLI)

Run the entire pipeline with one command:

python -m finance_analyzer analyze data/raw/my_export.csv \
    --rules config/categories_rules.json \
    --out reports/ \
    --top 15

✔ Streamlit Web App

Interactive dashboard:

python -m streamlit run streamlit_app.py


Features:

Upload CSV

Apply filters (year/month)

View categorized transactions

Income/expense metrics

Built-in charts

Download cleaned CSV

📂 Project Structure
fin_cont/
├── finance_analyzer/
│   ├── loader.py
│   ├── cleaner.py
│   ├── categorizer.py
│   ├── analytics.py
│   ├── visuals.py
│   ├── __main__.py      # CLI entry point
│
├── config/
│   └── categories_rules.json
│
├── data/
│   └── raw/             # (ignored in Git)
│
├── reports/             # generated plots (ignored in Git)
│
├── streamlit_app.py     # Streamlit UI
├── playground.py        # development sandbox
└── README.md

🔧 Installation
1. Create an environment (recommended)
conda create -n fin_cont_env python=3.11
conda activate fin_cont_env

2. Install dependencies
python -m pip install streamlit pandas matplotlib numba pyarrow


(You can later generate a requirements.txt.)

▶️ Usage
CLI
python -m finance_analyzer analyze \
    data/raw/your_export.csv \
    --rules config/categories_rules.json \
    --out reports \
    --top 20

Streamlit UI
python -m streamlit run streamlit_app.py


Visit: http://localhost:8501

🧠 Category Rules

Edit this file to improve classification:

config/categories_rules.json

Example:

{
  "Groceries": ["biedronka", "lidl", "zabka", "kaufland"],
  "Transport": ["uber", "bolt", "pkp"],
  "Shopping": ["allegro", "amazon"],
  "Salary": ["wynagrodzenie", "salary"]
}

📊 Example visualizations

(You can add your generated PNGs here later)

Monthly income vs expenses

Monthly category breakdown

Top expenses

🧩 Future improvements

Optional ideas:

Detect subscriptions / recurring payments

Forecast cashflow

Category editor in Streamlit

Export to PDF financial reports

Deploy Streamlit online (Streamlit Cloud, HuggingFace Spaces)

📜 License

MIT License

✍️ Author

Tomasz Młynik