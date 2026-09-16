import numpy as np
import pandas as pd
import warnings
import mlcroissant as mlc
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

# 1. Load Data
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset/croissant/download')
record_sets = croissant_dataset.metadata.record_sets
df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
df.columns = df.columns.str.replace('american_bankruptcy.csv/', '')

# 2. Rename columns
column_mapping = {
    'X1': 'current_assets', 'X2': 'cost_of_goods_sold', 'X3': 'depreciation_amortization',
    'X4': 'ebitda', 'X5': 'inventory', 'X6': 'net_income', 'X7': 'total_receivables',
    'X8': 'market_value', 'X9': 'net_sales', 'X10': 'total_assets', 'X11': 'total_longterm_debt',
    'X12': 'ebit', 'X13': 'gross_profit', 'X14': 'total_current_liabilities',
    'X15': 'retained_earnings', 'X16': 'total_revenue', 'X17': 'total_liabilities',
    'X18': 'total_operating_expenses'
}
df = df.rename(columns=column_mapping)

# 3. Feature Engineering
df['current_ratio'] = df['current_assets'] / df['total_current_liabilities'].replace(0, float('nan'))
df['working_capital_to_assets'] = (df['current_assets'] - df['total_current_liabilities']) / df['total_assets'].replace(0, float('nan'))
df['roa'] = df['net_income'] / df['total_assets'].replace(0, float('nan'))
df['ebitda_margin'] = df['ebitda'] / df['total_revenue'].replace(0, float('nan'))
df['gross_margin'] = df['gross_profit'] / df['total_revenue'].replace(0, float('nan'))
df['ebit_margin'] = df['ebit'] / df['total_revenue'].replace(0, float('nan'))
df['retained_earnings_to_assets'] = df['retained_earnings'] / df['total_assets'].replace(0, float('nan'))
df['debt_ratio'] = df['total_liabilities'] / df['total_assets'].replace(0, float('nan'))
df['longterm_debt_to_assets'] = df['total_longterm_debt'] / df['total_assets'].replace(0, float('nan'))
df['debt_to_equity'] = df['total_liabilities'] / (df['total_assets'] - df['total_liabilities']).replace(0, float('nan'))
df['asset_turnover'] = df['net_sales'] / df['total_assets'].replace(0, float('nan'))
df['inventory_turnover'] = df['cost_of_goods_sold'] / df['inventory'].replace(0, float('nan'))
df['receivables_turnover'] = df['net_sales'] / df['total_receivables'].replace(0, float('nan'))
df['market_to_liabilities'] = df['market_value'] / df['total_liabilities'].replace(0, float('nan'))
df['market_to_assets'] = df['market_value'] / df['total_assets'].replace(0, float('nan'))
df['log_total_assets'] = np.log(df['total_assets'].clip(lower=1e-5))

new_ratio_cols = ['current_ratio', 'working_capital_to_assets', 'roa', 'ebitda_margin',
                  'gross_margin', 'ebit_margin', 'retained_earnings_to_assets', 'debt_ratio',
                  'longterm_debt_to_assets', 'debt_to_equity', 'asset_turnover',
                  'inventory_turnover', 'receivables_turnover', 'market_to_liabilities', 'market_to_assets',
                  'log_total_assets']

for col in new_ratio_cols:
    df[col] = df[col].fillna(df[col].median())

df['bankrupt'] = (df['status_label'] == b'failed').astype(int)

# 4. Split
train = df[df['year'] <= 2011]
val   = df[(df['year'] >= 2012) & (df['year'] <= 2014)]
test  = df[df['year'] >= 2015]

drop_cols = ['company_name', 'status_label', 'year', 'bankrupt']
if 'period' in train.columns:
    drop_cols.append('period')

X_train_raw = train.drop(columns=drop_cols)
y_train_raw = train['bankrupt']
X_val  = val.drop(columns=drop_cols)
y_val  = val['bankrupt']
X_test = test.drop(columns=drop_cols)
y_test = test['bankrupt']

# 5. Under-sample
rus = RandomUnderSampler(sampling_strategy=0.4286, random_state=42)
X_train, y_train = rus.fit_resample(X_train_raw, y_train_raw)

# 6. Winsorize
class Winsorizer:
    def __init__(self, lower_pct=0.01, upper_pct=0.01):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bounds = {}
    def fit(self, df, columns):
        for col in columns:
            self.bounds[col] = (df[col].quantile(self.lower_pct), df[col].quantile(1.0 - self.upper_pct))
    def transform(self, df, columns):
        df_copy = df.copy()
        for col in columns:
            lower, upper = self.bounds[col]
            df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
        return df_copy

winsorizer = Winsorizer(lower_pct=0.01, upper_pct=0.01)
winsorizer.fit(X_train_raw, new_ratio_cols)
X_train_winsor = winsorizer.transform(X_train, new_ratio_cols)
X_val_winsor = winsorizer.transform(X_val, new_ratio_cols)
X_test_winsor = winsorizer.transform(X_test, new_ratio_cols)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_winsor[new_ratio_cols])
X_val_scaled = scaler.transform(X_val_winsor[new_ratio_cols])
X_test_scaled = scaler.transform(X_test_winsor[new_ratio_cols])

# 7. Evaluate Baseline Models
# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_val_pred_rf = rf.predict_proba(X_val_scaled)[:, 1]
y_test_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]

# XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
xgb.fit(X_train_scaled, y_train)
y_val_pred_xgb = xgb.predict_proba(X_val_scaled)[:, 1]
y_test_pred_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

# Show AUC metrics
print("\n--- Model Performance Comparison ---")
for name, val_preds, test_preds in [("RF", y_val_pred_rf, y_test_pred_rf), ("XGB", y_val_pred_xgb, y_test_pred_xgb)]:
    val_roc = roc_auc_score(y_val, val_preds)
    val_pr = average_precision_score(y_val, val_preds)
    test_roc = roc_auc_score(y_test, test_preds)
    test_pr = average_precision_score(y_test, test_preds)
    print(f"{name}: Val ROC = {val_roc:.4f}, Val PR = {val_pr:.4f} | Test ROC = {test_roc:.4f}, Test PR = {test_pr:.4f}")

# Analyze precision/recall at different risk buckets
def analyze_risk_buckets(y_true, y_pred, name):
    eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    eval_df = eval_df.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    total_defaults = eval_df['y_true'].sum()
    n_obs = len(eval_df)
    
    print(f"\nRisk Bucket Analysis for {name}:")
    for pct in [5, 10, 20]:
        cutoff = int(n_obs * (pct / 100))
        # flagged firms are those up to cutoff
        flagged = eval_df.iloc[:cutoff]
        tp = flagged['y_true'].sum()
        fp = cutoff - tp
        recall = tp / total_defaults * 100
        precision = tp / cutoff * 100
        print(f"  Top {pct}% bucket ({cutoff} firms flagged): {tp} defaults caught out of {total_defaults} total. Recall = {recall:.2f}%, Precision = {precision:.2f}%")

analyze_risk_buckets(y_test, y_test_pred_rf, "Random Forest")
analyze_risk_buckets(y_test, y_test_pred_xgb, "XGBoost")
