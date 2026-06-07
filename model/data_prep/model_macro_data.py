import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load your individual individual player baseline matrix
df_players = pd.read_csv("../../data/training/nil_data.csv")

# 2. LOAD YOUR SECONDARY FILE (Simulating your School/Sport macro file layout)
# Substitute this block with your actual found csv data frames!
unique_schools = df_players['School'].unique()
macro_school_data = []

np.random.seed(42)
for school in unique_schools:
    # Simulating your macro metrics (e.g. Total Department Football Revenue / General Social Footprint)
    macro_school_data.append({
        "School": school,
        "Sport": "Football",
        "School_Athletic_Revenue_M": np.random.uniform(40, 180),
        "School_Total_Sport_Followers_K": np.random.uniform(100, 2500)
    })
df_macro = pd.DataFrame(macro_school_data)

# 3. PANDAS INNER-JOIN: Merge the macro feature arrays into the individual matrix
# This maps the school's overall economic weight to every individual player row
df_merged = pd.merge(df_players, df_macro, on=["School"], how="left")

# 4. ADVANCED FEATURE ENGINEERING: Hierarchical Baseline Target Encoding
# Calculate the average position multipliers based on historical sports economics
pos_multipliers = {'QB': 2.2, 'WR': 1.6, 'EDGE': 1.5, 'OT': 1.4, 'CB': 1.1, 'DL': 1.0, 'LB': 0.8, 'RB': 0.7, 'S': 0.6,
                   'TE': 0.5, 'IOL': 0.4}
df_merged['Position_Weight'] = df_merged['Position'].map(pos_multipliers).fillna(0.5)

# Calculate a compound proxy: Revenue * Position Weight * Star Tier
df_merged['Star_Rating'] = 5.0 - (df_merged['Rank'] * 0.025)
df_merged['Star_Rating'] = df_merged['Star_Rating'].clip(lower=3.0)

df_merged['Engineered_Market_Anchor'] = (
        df_merged['School_Total_Sport_Followers_K'] *
        df_merged['Position_Weight'] *
        df_merged['Star_Rating']
)

# 5. Compile Feature Matrix
X_features = df_merged[
    ['School_Athletic_Revenue_M', 'School_Total_Sport_Followers_K', 'Star_Rating', 'Engineered_Market_Anchor']]
y_nil = df_merged['NIL_Valuation'].values
y_log = np.log10(y_nil)  # Smooth out exponential skew anomalies

# 6. Out-of-Sample Cross-Validation Evaluation Loop
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions_log = np.zeros(len(df_merged))

for train_idx, test_idx in kf.split(X_features):
    X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
    y_train, y_test = y_log[train_idx], y_log[test_idx]

    # Random Forest effortlessly blends macro school economics with player tiers
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    oof_predictions_log[test_idx] = model.predict(X_test)

# 7. Reverse transform target arrays back to true USD scales
y_pred_dollars = 10 ** oof_predictions_log
final_test_r2 = r2_score(y_nil, y_pred_dollars)
final_mae = mean_absolute_error(y_nil, y_pred_dollars)

print("--- HIERARCHICAL SCHOOL-TIER PERFORMANCE ---")
print(f"Verified Testing R² Score: {final_test_r2:.4f}")
print(f"True Mean Absolute Error : ${final_mae:,.2f}")
