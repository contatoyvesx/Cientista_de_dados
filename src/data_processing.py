from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "credit_default_raw.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "credit_default_processed.csv"


def load_data() -> pd.DataFrame:
    """Carrega os dados brutos da base de crédito."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_DATA_FILE}. "
            "Execute primeiro src/data_collection.py."
        )

    return pd.read_csv(RAW_DATA_FILE)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza tratamentos básicos para preparar a base para análise e modelagem."""
    data = df.copy()

    # Remove registros duplicados.
    data = data.drop_duplicates().reset_index(drop=True)

    # A base utiliza valores negativos em algumas variáveis de histórico de pagamento.
    # Mantemos os valores nesta etapa para não alterar o significado original dos dados.

    # Garante que o target seja inteiro.
    data["default"] = data["default"].astype(int)

    return data


def save_processed_data(df: pd.DataFrame) -> None:
    """Salva a base tratada em data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Dados processados salvos em: {PROCESSED_DATA_FILE}")


def main() -> None:
    print("Carregando dados...")
    df = load_data()

    print(f"Formato original: {df.shape}")
    print(f"Duplicatas: {df.duplicated().sum()}")
    print(f"Valores ausentes: {df.isna().sum().sum()}")

    processed_df = process_data(df)

    print(f"Formato após tratamento: {processed_df.shape}")
    print("\nTipos de dados:")
    print(processed_df.dtypes)

    save_processed_data(processed_df)


if __name__ == "__main__":
    main()
