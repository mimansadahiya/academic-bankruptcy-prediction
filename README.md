# U.S. Corporate Default Prediction

Predictive modeling project to identify which U.S. public companies are likely to enter bankruptcy in the subsequent fiscal year.

## Dataset

8,262 companies across 78,682 firm-year observations (1999–2018) from the [US Company Bankruptcy Prediction Dataset](https://www.kaggle.com/) on Kaggle.

## Approach

Three-phase credit risk modeling:

1. **Altman-Style Benchmark** — Accounting-based scoring model using standard financial ratios
2. **Logistic Regression** — Statistical interpretation of key financial drivers (profitability, leverage, size, liquidity)
3. **Machine Learning** — Random Forests, XGBoost, and Neural Networks to capture non-linear interactions

## Key Results

| Metric | Baseline Model | LASSO Model |
|--------|:-:|:-:|
| Test ROC-AUC | **0.7367** | 0.6992 |
| Top 5% Risk Bucket Recall | 14.29% | **21.60%** |
| Top 20% Risk Bucket Recall | **56.45%** | 45.99% |

## Feature Engineering

15 financial ratios across five dimensions:
- **Liquidity:** Current Ratio, Working Capital to Assets
- **Profitability:** ROA, EBITDA Margin, Gross Margin, Retained Earnings to Assets
- **Leverage:** Debt Ratio, Long-Term Debt to Assets, Debt-to-Equity
- **Efficiency:** Asset Turnover, Inventory Turnover, Receivables Turnover
- **Market Valuation:** Market Value to Liabilities, Market Value to Assets

## Tech Stack

- Python
- scikit-learn
- Jupyter Notebook
- Pandas, NumPy
