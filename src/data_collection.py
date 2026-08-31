from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "credit_default_raw.csv"


def collect_credit_data() -> pd.DataFrame:
    """Baixa e salva a base Default of Credit Card Clients (UCI)."""
    print("Baixando dataset UCI...")

    dataset = fetch_ucirepo(id=350)

    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()

    target_column = target.columns[0]
    target = target.rename(columns={target_column: "default"})

    data = pd.concat([features, target], axis=1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset salvo em: {OUTPUT_FILE}")
    print(f"Linhas: {data.shape[0]}")
    print(f"Colunas: {data.shape[1]}")

    return data


if __name__ == "__main__":
    df = collect_credit_data()

    print("\nPrimeiras 5 linhas:")
    print(df.head())

    print("\nDistribuição do target:")
    print(df["default"].value_counts())
