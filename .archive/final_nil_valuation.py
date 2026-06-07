import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. SETUP YOUR RETRAINED PRODUCTION MODEL
# ==========================================
# (Using the variable names from your successful wide-matrix join script)
X_categorical = pd.get_dummies(df_final_ml[['Position']], drop_first=True).astype(float)
X_numerical = df_final_ml[['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL', 'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']]
X_train_final = pd.concat([X_numerical, X_categorical], axis=1)

y_nil = df_final_ml['NIL_Valuation'].values
y_log = np.log10(y_nil)

# Train the final absolute model across all your high-density historical data
production_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
production_model.fit(X_train_final, y_log)


# ==========================================
# 2. INGEST & PREPROCESS THE FSU ROSTER
# ==========================================
# Assuming 'df_fsu_roster' is your already loaded Florida State roster dataframe.
# It must contain at least 'Player' and 'Position' columns.
# Example if needed: df_fsu_roster = pd.read_csv("fsu_roster.csv")

# Inject FSU's specific macro platform clout metrics across the entire roster row space
# (Using the verified FSU metrics we compiled earlier)
df_fsu_roster['FACEBOOK_FOOTBALL'] = 0.0      # Insert exact values from your xlsx if present
df_fsu_roster['INSTAGRAM_FOOTBALL'] = 0.0     # Insert exact values from your xlsx if present
df_fsu_roster['TIKTOK_FOOTBALL'] = 0.0        # Insert exact values from your xlsx if present
df_fsu_roster['TWITTER_FOOTBALL'] = 450000.0   # Example placeholder
df_fsu_roster['YOUTUBE_FOOTBALL'] = 55000.0    # Example placeholder

# One-hot encode the roster positions
X_fsu_categorical = pd.get_dummies(df_fsu_roster[['Position']], drop_first=True).astype(float)
X_fsu_numerical = df_fsu_roster[['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL', 'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']]
X_fsu_encoded = pd.concat([X_fsu_numerical, X_fsu_categorical], axis=1)

# CRITICAL DATA SCIENCE ALIGNMENT:
# Ensure the columns on the live roster match the training matrix exactly.
# This prevents crashes if the roster contains positions your model hasn't seen yet!
X_fsu_aligned = X_fsu_encoded.reindex(columns=X_train_final.columns, fill_value=0.0)


# ==========================================
# 3. GENERATE ROSTER-WIDE VALUATIONS
# ==========================================
# Predict log scale metrics
fsu_preds_log = production_model.predict(X_fsu_aligned)

# Invert log transform back to true dollar values
df_fsu_roster['Predicted_NIL_Valuation'] = 10 ** fsu_preds_log

# Format the final dataset cleanly for display/export
df_fsu_roster['Predicted_NIL_Valuation_Formatted'] = df_fsu_roster['Predicted_NIL_Valuation'].map(lambda x: f"${x:,.2f}")

# Save the complete roster evaluations out to a clean spreadsheet file
df_fsu_roster[['Player', 'Position', 'Predicted_NIL_Valuation_Formatted']].to_csv("fsu_predicted_valuations.csv", index=False)

print("\n==========================================")
print("   ROSTER EVALUATION ENGINE SUCCESS       ")
print("==========================================")
print(df_fsu_roster[['Player', 'Position', 'Predicted_NIL_Valuation_Formatted']].head(15).to_string(index=False))
