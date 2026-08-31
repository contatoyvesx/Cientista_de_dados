from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "credit_default_processed.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "credit_default_features.csv"


BILL_COLUMNS = ["X12", "X13", "X14", "X15", "X16", "X17"]
PAYMENT_COLUMNS = ["X18", "X19", "X20", "X21", "X22", "X23"]
PAYMENT_HISTORY_COLUMNS = ["X6", "X7", "X8", "X9", "X10", "X11"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-oriented features for credit-risk modeling."""
    data = df.copy()

    # Aggregate recent billing and payment behavior.
    data["avg_bill"] = data[BILL_COLUMNS].mean(axis=1)
    data["avg_payment"] = data[PAYMENT_COLUMNS].mean(axis=1)
    data["total_bill"] = data[BILL_COLUMNS].sum(axis=1)
    data["total_payment"] = data[PAYMENT_COLUMNS].sum(axis=1)

    # Avoid unstable ratios when the denominator is zero or negative.
    positive_bill = data["avg_bill"].clip(lower=1)
    data["payment_ratio"] = (
        data["avg_payment"].clip(lower=0) / positive_bill
    ).clip(upper=1.0)

    # Payment behavior over the six observed months.
    data["payment_trend"] = (
        data["avg_payment"] - data[PAYMENT_COLUMNS[0]]
    )

    # Recent payment-status behavior. Positive values indicate worse
    # payment-status codes in this dataset.
    data["avg_payment_status"] = data[PAYMENT_HISTORY_COLUMNS].mean(axis=1)
    data["max_payment_status"] = data[PAYMENT_HISTORY_COLUMNS].max(axis=1)
    data["recent_payment_status"] = data[PAYMENT_HISTORY_COLUMNS[0]]
    data["late_payment_months"] = (data[PAYMENT_HISTORY_COLUMNS] > 0).sum(axis=1)

    # Credit utilization proxy based on average bill and credit limit.
    credit_limit = data["X1"].clip(lower=1)
    data["avg_bill_to_limit"] = (
        data["avg_bill"].clip(lower=0) / credit_limit
    ).clip(upper=1.0)

    # Remove non-finite values produced by unusual source observations.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)

    return data


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    print("Carregando dados processados...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Linhas: {len(df)}")
    print(f"Colunas originais: {len(df.columns)}")

    features_df = create_features(df)
    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Features criadas: {len(features_df.columns) - len(df.columns)}")
    print(f"Dados com features salvos em: {OUTPUT_FILE}")
    print("\nNovas features:")
    new_columns = [column for column in features_df.columns if column not in df.columns]
    print(new_columns)
    print("\nAmostra:")
    print(features_df[new_columns + ["default"]].head())


if __name__ == "__main__":
    main()
