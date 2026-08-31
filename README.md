# Credit Risk & Pricing

Projeto de Ciência de Dados aplicado a risco de crédito e pricing.

## Objetivo

Estimar a probabilidade de inadimplência (PD) e utilizar o risco estimado em uma simulação de pricing de crédito.

## Pipeline

API / Dataset → Tratamento → EDA → Feature Engineering → Modelo → Avaliação → PD → Expected Loss → Pricing

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- SQL
- API REST
- Matplotlib / Seaborn

## Estrutura

```text
credit-risk-pricing/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── api_bcb.py
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── pricing.py
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── requirements.txt
└── README.md
```

## Fonte dos dados

Base Default of Credit Card Clients, UCI Machine Learning Repository.
