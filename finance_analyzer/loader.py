import pandas as pd

def load_transactions(csv_path: str) -> pd.DataFrame:
    encodings_to_try = ["cp1250", "iso-8859-2", "utf-8"]
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                csv_path,
                sep=None,
                engine='python',
                encoding=encoding,
            )
            print(f"Successfully loaded with encoding: {encoding}")
            return df
        except Exception:
            pass

    raise ValueError("Failed to load CSV with tried encodings.")