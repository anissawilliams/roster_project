from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df_master_ml = pd.read_csv("../data/training/master_ml_data.csv")

# 1. Process Feature Space
X_categorical = pd.get_dummies(df_master_ml[['position']], drop_first=True).astype(float)
X_numerical = df_master_ml[['weight', 'height', 'year', 'stars', 'rating', 'social_followers', 'Is_Transfer']]
X_train = pd.concat([X_numerical, X_categorical], axis=1)

y_nil = df_master_ml['nil_value'].values
y_log = np.log10(y_nil)

# 2. Cross-Validation Loop
loo = LeaveOneOut()
oof_preds_log = np.zeros(len(df_master_ml))

for train_idx, test_idx in loo.split(X_train):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_log[train_idx], y_log[test_idx]

    # max_depth=3 prevents overfitting on your high-signal custom rows
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    oof_preds_log[test_idx] = model.predict(X_te)

# 3. Invert log transformations and print results
y_pred_dollars = 10 ** oof_preds_log
print("\n==========================================")
print(f"Roster Model Testing R²: {r2_score(y_nil, y_pred_dollars):.4f}")
print(f"Roster Model MAE       : ${mean_absolute_error(y_nil, y_pred_dollars):,.2f}")
