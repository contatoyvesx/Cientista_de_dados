from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "credit_default_features.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "credit_risk_model.joblib"

TARGET = "default"


def train_model(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    return model, X.columns.tolist()


def simulate_client(model, feature_columns):
    """Interactive simulation using the same columns used during training."""
    print("\n=== Simulação de novo cliente ===")
    print("Informe os dados básicos do cliente.\n")

    credit_limit = float(input("Limite de crédito (X1): R$ "))
    age = int(input("Idade (X5): "))

    recent_status = int(input("Status de pagamento mais recente (X6): "))
    previous_status = int(input("Status de pagamento anterior (X7): "))
    older_status = int(input("Status de pagamento há 2 meses (X8): "))

    avg_bill = float(input("Fatura média dos últimos meses: R$ "))
    avg_payment = float(input("Pagamento médio dos últimos meses: R$ "))

    payment_ratio = min(max(avg_payment / max(avg_bill, 1), 0), 1)
    late_payment_months = int(
        input("Quantidade de meses com atraso (0 a 6): ")
    )

    client = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # Original variables not explicitly collected remain at zero for this
    # quick interview/demo simulation. The engineered variables are filled
    # from the information supplied by the user.
    client["X1"] = credit_limit
    client["X5"] = age
    client["X6"] = recent_status
    client["X7"] = previous_status
    client["X8"] = older_status

    client["avg_bill"] = avg_bill
    client["avg_payment"] = avg_payment
    client["total_bill"] = avg_bill * 6
    client["total_payment"] = avg_payment * 6
    client["payment_ratio"] = payment_ratio
    client["payment_trend"] = 0
    client["avg_payment_status"] = (
        recent_status + previous_status + older_status
    ) / 3
    client["max_payment_status"] = max(
        recent_status, previous_status, older_status
    )
    client["recent_payment_status"] = recent_status
    client["late_payment_months"] = late_payment_months
    client["avg_bill_to_limit"] = min(
        max(avg_bill / max(credit_limit, 1), 0), 1
    )

    probability = model.predict_proba(client[feature_columns])[0, 1]

    if probability < 0.20:
        risk = "BAIXO"
    elif probability < 0.50:
        risk = "MÉDIO"
    else:
        risk = "ALTO"

    print("\n=== Resultado ===")
    print(f"Probabilidade estimada de default: {probability:.2%}")
    print(f"Classificação de risco: {risk}")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset com features não encontrado: {DATA_FILE}"
        )

    print("Carregando dataset com features...")
    df = pd.read_csv(DATA_FILE)

    model, feature_columns = train_model(df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": feature_columns},
        MODEL_FILE,
    )

    print(f"\nModelo salvo em: {MODEL_FILE}")
    simulate_client(model, feature_columns)


if __name__ == "__main__":
    main()
