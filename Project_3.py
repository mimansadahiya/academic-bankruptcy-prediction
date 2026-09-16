#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 3: Corporate Default Prediction
Combined Script reproducing all results.
"""

def display(*args):
    for arg in args:
        print(arg)


# ==============================================================================
# # **Loading The Data**
# ==============================================================================

# ── Standard Library ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlcroissant as mlc
warnings.filterwarnings('ignore')

# ── Sklearn: Preprocessing & Model Selection ───────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# ── Sklearn: Models ────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ── XGBoost ────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── Neural Network ─────────────────────────────────────────────
from sklearn.neural_network import MLPClassifier

# ── Imbalanced Learn (optional) ────────────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Fetch the Croissant JSON-LD
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset/croissant/download')

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets
print(record_sets)

# Fetch the records and put them in a DataFrame
df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
df.head()


# remove the 'american_bankruptcy.csv/' prefix on the column names
df.columns = df.columns.str.replace('american_bankruptcy.csv/', '')
display(df.head())


# ==============================================================================
# # **Data Description**
# 
# ### **Unit of Analysis**
# Each unique observation consists of a company and the corresponding reporting year.
# 
# ### **Features**
# 
# - **company_name**
# - **status_label -** Company Status (Target Column)
# - **year**
# - **X1: Current assets -** All the assets of a company that are expected to be sold or used as a result of standard business
# - **X2: Cost of goods sold -** The total amount a company paid as a cost directly related to the sale of products
# - **X3: Depreciation and amortization -** Depreciation refers to the loss of value of a tangible fixed asset oveR
# - **X4: EBITDA -** Earnings before interest, taxes, depreciation, and amortization. A measure of a company’s overall
# - **X5: Inventory -** The accounting of items and raw materials that a company either uses in production or sells
# - **X6: Net Income -** The overall profitability of a company after all expenses and costs have been deducted from total revenue
# - **X7: Total Receivables -** The balance of money due to a firm for goods or services delivered or used but not yet paid for
# - **X8: Market Value -** The price of an asset in a marketplace. In our dataset, it refers to the market capitalization
# - **X9: Net Sales -** The sum of a company’s gross sales minus its returns, allowances, and discounts
# - **X10: Total Assets -** All the assets, or items of value, a business owns
# - **X11: Total Long-term Debt -** A company’s loans and other liabilities that will not become due within one year of the balance
# - **X12: EBIT -** Earnings before interest and taxes
# - **X13: Gross Profit -** The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services
# - **X14: Total Current Liabilities -** The sum of accounts payable, accrued liabilities, and taxes such as bonds payable at the end of the year, salaries, and commissions remaining
# - **X15: Retained Earnings -** The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and dividends to shareholders
# - **X16: Total Revenue -** The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments
# - **X17: Total Liabilities -** The combined debts and obligations that the company owes to outside parties
# - **X18: Total Operating Expenses -** The expense a business incurs through its normal business operations
# ==============================================================================

# rename the columns so we dont have to keep referring to the column dictionary above
column_mapping = {
    'X1': 'current_assets',
    'X2': 'cost_of_goods_sold',
    'X3': 'depreciation_amortization',
    'X4': 'ebitda',
    'X5': 'inventory',
    'X6': 'net_income',
    'X7': 'total_receivables',
    'X8': 'market_value',
    'X9': 'net_sales',
    'X10': 'total_assets',
    'X11': 'total_longterm_debt',
    'X12': 'ebit',
    'X13': 'gross_profit',
    'X14': 'total_current_liabilities',
    'X15': 'retained_earnings',
    'X16': 'total_revenue',
    'X17': 'total_liabilities',
    'X18': 'total_operating_expenses'
}

df = df.rename(columns=column_mapping)
df.head()


# ==============================================================================
# ### **Checking for Missing Values**
# ==============================================================================

# check columns for missing values - there are no missing values in any of the columns
print(df.isnull().sum())


# ==============================================================================
# ### **Checking for Extreme Values**
# ==============================================================================

# check columns for extreme values
df.describe().T

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

numeric_cols = df.select_dtypes(include='number').drop(columns=['year'])

fig, axes = plt.subplots(6, 3, figsize=(18, 24))
axes = axes.flatten()

for i, col in enumerate(numeric_cols.columns):
    axes[i].hist(df[col], bins=50, edgecolor='black', color='steelblue', alpha=0.7)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.suptitle('Distribution of All Financial Variables', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()


# ==============================================================================
# ### **Checking Other Data Features**
# ==============================================================================

# 1. Firm-year observations
print("=== FIRM-YEAR OBSERVATIONS ===")
print("Total firm-year obs:", len(df))
print("Unique firms:", df['company_name'].nunique())
print("Year range:", df['year'].min(), "to", df['year'].max())
print("Unique years:", df['year'].nunique())

# 2. Bankrupt observations
print("\n=== BANKRUPTCY COUNTS ===")
print(df['status_label'].value_counts())
print(df['status_label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# 3. Bankruptcy rate by sample period
print("\n=== BANKRUPTCY RATE BY PERIOD ===")
def get_period(year):
    if year <= 2011:
        return 'Train (1999-2011)'
    elif year <= 2014:
        return 'Validation (2012-2014)'
    else:
        return 'Test (2015-2018)'

df['period'] = df['year'].apply(get_period)
period_stats = df.groupby('period')['status_label'].apply(
    lambda x: f"{(x == b'failed').sum() / len(x) * 100:.2f}%"
)
print(period_stats)


# ==============================================================================
# # **Feature Engineering**
# ==============================================================================

# Avoid division by zero by replacing 0s with NaN
df['current_ratio'] = df['current_assets'] / df['total_current_liabilities'].replace(0, float('nan'))
df['working_capital_to_assets'] = (df['current_assets'] - df['total_current_liabilities']) / df['total_assets'].replace(0, float('nan'))

# Profitability
df['roa'] = df['net_income'] / df['total_assets'].replace(0, float('nan'))
df['ebitda_margin'] = df['ebitda'] / df['total_revenue'].replace(0, float('nan'))
df['gross_margin'] = df['gross_profit'] / df['total_revenue'].replace(0, float('nan'))
df['ebit_margin'] = df['ebit'] / df['total_revenue'].replace(0, float('nan'))
df['retained_earnings_to_assets'] = df['retained_earnings'] / df['total_assets'].replace(0, float('nan'))

# Leverage
df['debt_ratio'] = df['total_liabilities'] / df['total_assets'].replace(0, float('nan'))
df['longterm_debt_to_assets'] = df['total_longterm_debt'] / df['total_assets'].replace(0, float('nan'))
df['debt_to_equity'] = df['total_liabilities'] / (df['total_assets'] - df['total_liabilities']).replace(0, float('nan'))

# Efficiency
df['asset_turnover'] = df['net_sales'] / df['total_assets'].replace(0, float('nan'))
df['inventory_turnover'] = df['cost_of_goods_sold'] / df['inventory'].replace(0, float('nan'))
df['receivables_turnover'] = df['net_sales'] / df['total_receivables'].replace(0, float('nan'))

# Market-based
df['market_to_liabilities'] = df['market_value'] / df['total_liabilities'].replace(0, float('nan'))
df['market_to_assets'] = df['market_value'] / df['total_assets'].replace(0, float('nan'))


# Add size metric
df['log_total_assets'] = np.log(df['total_assets'].clip(lower=1e-5))

# ALTMAN

df['altman_X1'] = (df['current_assets'] - df['total_current_liabilities']) / df['total_assets'].replace(0, float('nan'))
df['altman_X2'] = df['retained_earnings'] / df['total_assets'].replace(0, float('nan'))
df['altman_X3'] = df['ebit'] / df['total_assets'].replace(0, float('nan'))
df['altman_X4'] = df['market_value'] / df['total_liabilities'].replace(0, float('nan'))
df['altman_X5'] = df['net_sales'] / df['total_assets'].replace(0, float('nan'))

df['altman_z'] = (1.2 * df['altman_X1'] +
                  1.4 * df['altman_X2'] +
                  3.3 * df['altman_X3'] +
                  0.6 * df['altman_X4'] +
                  1.0 * df['altman_X5'])

print("New columns added:", df.shape[1], "total columns")
df.head()

new_ratio_cols = ['current_ratio', 'working_capital_to_assets', 'roa', 'ebitda_margin',
                  'gross_margin', 'ebit_margin', 'retained_earnings_to_assets', 'debt_ratio',
                  'longterm_debt_to_assets', 'debt_to_equity', 'asset_turnover',
                  'inventory_turnover', 'receivables_turnover', 'market_to_liabilities', 'market_to_assets',
                  'log_total_assets']

print(df[new_ratio_cols].isnull().sum())

# we can either fill the null columns with 0 or median values - for now we fill with median values to avoid distortion
for col in new_ratio_cols:
    df[col] = df[col].fillna(df[col].median())


# ==============================================================================
# # **Modeling**
# ==============================================================================


# ==============================================================================
# ### **Splitting Data**
# ==============================================================================

df['bankrupt'] = (df['status_label'] == b'failed').astype(int)

# TRAIN / VAL / TEST SPLIT
train = df[df['year'] <= 2011]
val   = df[(df['year'] >= 2012) & (df['year'] <= 2014)]
test  = df[df['year'] >= 2015]

# RAW FEATURE MATRICES
drop_cols = ['company_name', 'status_label', 'year', 'period', 'bankrupt']

X_train_raw = train.drop(columns=drop_cols)
y_train_raw = train['bankrupt']

X_val  = val.drop(columns=drop_cols)
y_val  = val['bankrupt']

X_test = test.drop(columns=drop_cols)
y_test = test['bankrupt']


# UNDERSAMPLE TRAINING SET ONLY (70/30)
# This is so that we even out the minority class with majority class without making fake data
rus = RandomUnderSampler(sampling_strategy=0.4286, random_state=42)
X_train, y_train = rus.fit_resample(X_train_raw, y_train_raw)

print("=== ORIGINAL TRAINING SET ===")
print(f"Total: {len(y_train_raw)} | Bankrupt: {y_train_raw.sum()} ({y_train_raw.mean()*100:.2f}%)")

print("\n=== RESAMPLED TRAINING SET (70/30) ===")
print(f"Total: {len(y_train)} | Bankrupt: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

print("\n=== VALIDATION SET (unchanged) ===")
print(f"Total: {len(y_val)} | Bankrupt: {y_val.sum()} ({y_val.mean()*100:.2f}%)")

print("\n=== TEST SET (unchanged) ===")
print(f"Total: {len(y_test)} | Bankrupt: {y_test.sum()} ({y_test.mean()*100:.2f}%)")


# ==============================================================================
# ### **Altman**
# ==============================================================================

altman_cols = ['altman_X1', 'altman_X2', 'altman_X3', 'altman_X4', 'altman_X5', 'altman_z', 'bankrupt']
altman_train = train[altman_cols]
altman_val = val[altman_cols]
altman_test = test[altman_cols]

# mean z-score by status
print("=== AVERAGE Z-SCORE BY STATUS (Test Set) ===")
print(altman_test.groupby('bankrupt')[['altman_X1', 'altman_X2', 'altman_X3',
                                        'altman_X4', 'altman_X5', 'altman_z']]
      .mean().round(4).T.rename(columns={0: 'Alive', 1: 'Bankrupt'}))

# auc scores
altman_roc = roc_auc_score(altman_test['bankrupt'], -altman_test['altman_z'])
altman_prauc = average_precision_score(altman_test['bankrupt'], -altman_test['altman_z'])

print(f"\nROC-AUC: {altman_roc:.4f}")
print(f"PR-AUC:  {altman_prauc:.4f}")

# confusion matrix
altman_test['altman_pred'] = (altman_test['altman_z'] < 1.81).astype(int)
cm = confusion_matrix(altman_test['bankrupt'], altman_test['altman_pred'])

TP = cm[1,1]; FP = cm[0,1]; FN = cm[1,0]; TN = cm[0,0]
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== CONFUSION MATRIX (Z < 1.81 = Distress) ===")
print(f"                  Actual Distress   Actual Healthy")
print(f"Pred Distress     TP = {TP:<10}    FP = {FP}")
print(f"Pred Healthy      FN = {FN:<10}    TN = {TN}")
print(f"\nAccuracy:  {(TP+TN)/(TP+FP+FN+TN)*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")

# risk buckets
altman_sorted = altman_test.sort_values('altman_z', ascending=True)
total_bankruptcies = altman_sorted['bankrupt'].sum()

print("\n=== TOP RISK BUCKET RECALL ===")
for pct in [0.05, 0.10]:
    n = int(len(altman_sorted) * pct)
    captured = altman_sorted.head(n)['bankrupt'].sum()
    recall_b = captured / total_bankruptcies * 100
    print(f"Top {int(pct*100)}% ({n} firms): {captured} captured = {recall_b:.1f}% of all bankruptcies")

# zone breakdown
altman_test['zone'] = pd.cut(altman_test['altman_z'],
                              bins=[-float('inf'), 1.81, 2.99, float('inf')],
                              labels=['Distress (<1.81)', 'Grey (1.81-2.99)', 'Safe (>2.99)'])
zone_table = altman_test.groupby('zone')['bankrupt'].agg(['count', 'sum', 'mean'])
zone_table.columns = ['total_firms', 'bankruptcies', 'bankruptcy_rate']
zone_table['bankruptcy_rate'] = zone_table['bankruptcy_rate'].mul(100).round(2).astype(str) + '%'
print("\n=== Z-SCORE ZONE BREAKDOWN ===")
print(zone_table)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# histogram
axes[0].hist(altman_test[altman_test['bankrupt']==0]['altman_z'].clip(-50, 50),
             bins=50, alpha=0.6, label='Alive', color='steelblue')
axes[0].hist(altman_test[altman_test['bankrupt']==1]['altman_z'].clip(-50, 50),
             bins=50, alpha=0.6, label='Bankrupt', color='red')
axes[0].axvline(1.81, color='black',  linestyle='--', linewidth=1, label='Distress < 1.81')
axes[0].axvline(2.99, color='orange', linestyle='--', linewidth=1, label='Safe > 2.99')
axes[0].set_title('Z-Score Distribution by Status')
axes[0].set_xlabel('Z-Score')
axes[0].legend()

# scatterplot
axes[1].scatter(altman_test[altman_test['bankrupt']==0].index,
                altman_test[altman_test['bankrupt']==0]['altman_z'].clip(-50, 50),
                alpha=0.3, s=10, color='steelblue', label='Alive')
axes[1].scatter(altman_test[altman_test['bankrupt']==1].index,
                altman_test[altman_test['bankrupt']==1]['altman_z'].clip(-50, 50),
                alpha=0.7, s=20, color='red', label='Bankrupt')
axes[1].axhline(1.81, color='black',  linestyle='--', linewidth=1, label='Distress < 1.81')
axes[1].axhline(2.99, color='orange', linestyle='--', linewidth=1, label='Safe > 2.99')
axes[1].set_title('Z-Score Scatterplot by Firm')
axes[1].set_xlabel('Firm Index')
axes[1].set_ylabel('Z-Score')
axes[1].legend()

plt.suptitle('Altman Z-Score — Separation of Bankrupt vs Alive Firms (Test Set)', fontsize=13)
plt.tight_layout()
plt.show()


# ==============================================================================
# # **Logistic Regression**

# ==============================================================================


# ==============================================================================
# ### **Outlier Winsorization & Feature Scaling**
# We implement a Winsorizer class to clip extreme outlier ratios (e.g. debt-to-equity with very small denominators) at the 1st and 99th percentiles. The clipping bounds are fit exclusively on the training set to prevent lookahead bias. We then scale features using  to ensure logistic regression converges properly.

# ==============================================================================

# Winsorizer Class
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

# Fit on training and apply to all splits
winsorizer = Winsorizer(lower_pct=0.01, upper_pct=0.01)
winsorizer.fit(X_train_raw, new_ratio_cols)

X_train_winsorized = winsorizer.transform(X_train, new_ratio_cols)
X_val_winsorized = winsorizer.transform(X_val, new_ratio_cols)
X_test_winsorized = winsorizer.transform(X_test, new_ratio_cols)

print("Winsorization complete on train, val, and test splits.")



# ==============================================================================
# ### **Baseline Logistic Regression (6 Motivated Variables)**
# We train an unweighted baseline model using 6 key variables motivated by economic and credit risk theory.

# ==============================================================================

baseline_features = [
    'working_capital_to_assets',
    'roa',
    'retained_earnings_to_assets',
    'debt_ratio',
    'market_to_liabilities',
    'log_total_assets'
]

# Scaling
scaler_base = StandardScaler()
X_train_base = scaler_base.fit_transform(X_train_winsorized[baseline_features])
X_val_base = scaler_base.transform(X_val_winsorized[baseline_features])
X_test_base = scaler_base.transform(X_test_winsorized[baseline_features])

# Fit Model
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train_base, y_train)

# Show Results
print(f"Baseline Intercept: {lr_base.intercept_[0]:.4f}\n")
base_coef_df = pd.DataFrame({
    'Feature': baseline_features,
    'Coefficient (Beta)': lr_base.coef_[0],
    'Odds Ratio (exp(Beta))': np.exp(lr_base.coef_[0])
}).sort_values(by='Coefficient (Beta)')
display(base_coef_df)



# ==============================================================================
# ### **Regularized Logistic Regression (LASSO)**
# We fit a regularized model using all 15 ratios (plus log size metric) and L1 (LASSO) penalization, tuning the regularization strength $ on the Validation set to maximize the **PR-AUC** metric.

# ==============================================================================

# Scaling all features
scaler_all = StandardScaler()
X_train_all = scaler_all.fit_transform(X_train_winsorized[new_ratio_cols])
X_val_all = scaler_all.transform(X_val_winsorized[new_ratio_cols])
X_test_all = scaler_all.transform(X_test_winsorized[new_ratio_cols])

# Tune C on validation set
c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
tuning_results = []

for C in c_values:
    lr_lasso = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=1000, random_state=42)
    lr_lasso.fit(X_train_all, y_train)
    y_val_pred = lr_lasso.predict_proba(X_val_all)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred)
    tuning_results.append({'C': C, 'Val ROC-AUC': val_roc_auc, 'Val PR-AUC': val_pr_auc})

tuning_df = pd.DataFrame(tuning_results)
display(tuning_df)

best_c = tuning_df.loc[tuning_df['Val PR-AUC'].idxmax(), 'C']
print(f"\nBest C selected based on Val PR-AUC: {best_c}")


# Fit final model using best C
lr_lasso_final = LogisticRegression(penalty='l1', C=best_c, solver='liblinear', max_iter=1000, random_state=42)
lr_lasso_final.fit(X_train_all, y_train)

# Show Coefficients
lasso_coef_df = pd.DataFrame({
    'Feature': new_ratio_cols,
    'Coefficient (Beta)': lr_lasso_final.coef_[0],
    'Odds Ratio (exp(Beta))': np.exp(lr_lasso_final.coef_[0])
}).sort_values(by='Coefficient (Beta)')

print(f"LASSO Intercept: {lr_lasso_final.intercept_[0]:.4f}\n")
display(lasso_coef_df)



# ==============================================================================
# ### **Model Comparison & Curve Visualizations**
# We evaluate both the Baseline and LASSO models on the unseen out-of-sample **Test Set**.

# ==============================================================================

# Predict probabilities on test set
y_test_pred_base = lr_base.predict_proba(X_test_base)[:, 1]
y_test_pred_lasso = lr_lasso_final.predict_proba(X_test_all)[:, 1]

# Calculate performance
summary_results = {
    'Model': ['Baseline Logistic', 'LASSO Logistic'],
    'Test ROC-AUC': [roc_auc_score(y_test, y_test_pred_base), roc_auc_score(y_test, y_test_pred_lasso)],
    'Test PR-AUC': [average_precision_score(y_test, y_test_pred_base), average_precision_score(y_test, y_test_pred_lasso)]
}
display(pd.DataFrame(summary_results))


# Plot Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curves
RocCurveDisplay.from_predictions(y_test, y_test_pred_base, ax=ax1, name='Baseline Logistic', color='orange')
RocCurveDisplay.from_predictions(y_test, y_test_pred_lasso, ax=ax1, name='LASSO Logistic', color='blue')
ax1.plot([0, 1], [0, 1], linestyle='--', color='grey')
ax1.set_title('ROC Curves (Out-of-Sample Test Set)')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')

# PR Curves
PrecisionRecallDisplay.from_predictions(y_test, y_test_pred_base, ax=ax2, name='Baseline Logistic', color='orange')
PrecisionRecallDisplay.from_predictions(y_test, y_test_pred_lasso, ax=ax2, name='LASSO Logistic', color='blue')
ax2.set_title('Precision-Recall Curves (Test Set)')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')

plt.tight_layout()
plt.show()



# ==============================================================================
# ### **Top-Risk Bucket Recall Analysis**
# Lenders and credit analysts want to know **how many bankruptcies can be caught by flagging the riskiest 5%, 10%, or 20% of firms**. Here we calculate the percentage of total bankruptcies captured in these top-risk buckets.

# ==============================================================================

def evaluate_risk_buckets(y_true, y_pred, model_name):
    eval_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    eval_df = eval_df.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    
    total_bankruptcies = eval_df['y_true'].sum()
    n_obs = len(eval_df)
    
    results = []
    for pct in [5, 10, 20]:
        cutoff = int(n_obs * (pct / 100))
        bucket_bankruptcies = eval_df.loc[:cutoff, 'y_true'].sum()
        recall = bucket_bankruptcies / total_bankruptcies
        results.append({
            'Model': model_name,
            'Bucket': f'Top {pct}% Riskiest',
            'Bankruptcies Captured': bucket_bankruptcies,
            'Total Bankruptcies': total_bankruptcies,
            'Recall (%)': recall * 100
        })
    return pd.DataFrame(results)

base_buckets = evaluate_risk_buckets(y_test, y_test_pred_base, 'Baseline Model')
lasso_buckets = evaluate_risk_buckets(y_test, y_test_pred_lasso, 'LASSO Model')
display(pd.concat([base_buckets, lasso_buckets]).reset_index(drop=True))

# Grouped Bar Chart for Bankruptcy Recall by Risk Bucket
buckets = ['Top 5% Riskiest', 'Top 10% Riskiest', 'Top 20% Riskiest']
baseline_recall = [11.50, 32.75, 58.89]
lasso_recall = [21.95, 31.71, 47.74]

x = np.arange(len(buckets))
width = 0.35

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

baseline_color = '#4682B4'
lasso_color = '#E69F00'

rects1 = ax.bar(x - width/2, baseline_recall, width, label='Baseline Model', color=baseline_color, edgecolor='none')
rects2 = ax.bar(x + width/2, lasso_recall, width, label='LASSO Model', color=lasso_color, edgecolor='none')

ax.set_ylabel('Recall (%)', fontsize=12, fontweight='semibold', labelpad=10)
ax.set_xlabel('Risk Bucket (Firms ranked by predicted risk)', fontsize=12, fontweight='semibold', labelpad=10)
ax.set_title('Bankruptcy Recall by Risk Bucket (Out-of-Sample Test Set)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(buckets, fontsize=11)
ax.set_ylim(0, 100)
ax.set_xlim(-0.6, 3.4)

ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#cccccc')
ax.grid(False, axis='x')
ax.set_axisbelow(True)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Annotations (Relative Recall Increases)
ax.annotate('+91% Recall Increase', 
            xy=(0 + width/2, 11.0), xytext=(-0.45, 38),
            arrowprops=dict(facecolor='darkred', edgecolor='darkred', shrink=0.08, width=1.5, headwidth=6, headlength=6),
            fontsize=10, fontweight='bold', color='darkred')

ax.annotate('+23% Recall Increase', 
            xy=(2 - width/2, 30.0), xytext=(1.0, 72),
            arrowprops=dict(facecolor='darkgreen', edgecolor='darkgreen', shrink=0.08, width=1.5, headwidth=6, headlength=6),
            fontsize=10, fontweight='bold', color='darkgreen')

# Key Takeaways Box
takeaway_text = (
    "  KEY TAKEAWAYS  \n"
    "• Top 5% (High-Precision):\n"
    "  LASSO catches 91% more defaults\n"
    "  (22.0% vs 11.5%)\n\n"
    "• Top 20% (Broad Screening):\n"
    "  Baseline catches 23% more defaults\n"
    "  (58.9% vs 47.7%)\n\n"
    "• Top 10%:\n"
    "  Both models perform similarly\n"
    "  (~32-33%)"
)

props = dict(boxstyle='round,pad=0.8', facecolor='#fbfbfb', edgecolor='#d0d0d0', alpha=0.95)
ax.text(2.4, 90, takeaway_text, fontsize=9.5, bbox=props, va='top', ha='left', linespacing=1.3)

# Legend
ax.legend(title='Model', title_fontsize='11', fontsize='10', loc='upper left', frameon=True, shadow=False, facecolor='#ffffff')

plt.tight_layout()
plt.savefig('risk_bucket_recall.png', dpi=300)
plt.show()
