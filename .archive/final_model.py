import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load your newly integrated 25-row training matrix
# (Preserving your exact terminal structure layout)
df_final_training_matrix = pd.read_csv("../data/training/final_training_matrix.csv")
df = df_final_training_matrix  # Uses the variable from your successful join pipeline

# 2. Extract multi-platform continuous social metrics and position features
X_categorical = pd.get_dummies(df[['Position']], drop_first=True).astype(float)
X_numerical = df[['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL', 'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']]
X_final = pd.concat([X_numerical, X_categorical], axis=1)

y_nil = df['NIL_Valuation'].values
y_log = np.log10(y_nil)  # Essential log-scaling step to balance quarterback outliers

# 3. LOOCV (Leave-One-Out Cross-Validation) for Small-Sample Datasets (N=25)
# This trains on 24 rows and tests on the remaining 1 row, repeating 25 times
loo = LeaveOneOut()
oof_preds_log = np.zeros(len(df))
feature_importances = []

for train_idx, test_idx in loo.split(X_final):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y_log[train_idx], y_log[test_idx]

    # Restricting tree depth keeps the model from overfitting on 25 observations
    model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    oof_preds_log[test_idx] = model.predict(X_test)
    feature_importances.append(model.feature_importances_)

# 4. Invert log scales back into true market dollar values
y_pred_dollars = 10 ** oof_preds_log
final_test_r2 = r2_score(y_nil, y_pred_dollars)
final_mae = mean_absolute_error(y_nil, y_pred_dollars)

print("\n==========================================")
print("     FINAL PRODUCTION MODEL RESULTS       ")
print("==========================================")
print(f"Testing R² Score   : {final_test_r2:.4f}")
print(f"Mean Absolute Error: ${final_mae:,.2f}")

# 5. Extract the explicit platform drivers
avg_importance = np.mean(feature_importances, axis=0)
importance_df = pd.DataFrame({
    'Engineered_Feature': X_final.columns,
    'Clout_Importance_Weight': avg_importance
}).sort_values(by='Clout_Importance_Weight', ascending=False)

print("\n--- PLATFORM PREDICTIVE IMPORTANCE RANKINGS ---")
print(importance_df.to_string(index=False))
