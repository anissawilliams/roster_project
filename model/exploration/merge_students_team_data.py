import pandas as pd

# 1. Load both text files
df_players = pd.read_csv("../../data/training/nil_data.csv")
df_macro = pd.read_csv("../../data/training/school_macro_data.csv")

# 2. Execute a clean relational join on the School column
# This fixes the missing data and gives every player row its school's financial power metrics
df_final_ml = pd.merge(df_players, df_macro, on="School", how="left")

# 3. Fill any missing school matches with standard column averages (avoids NaN model crashes)
df_final_ml['School_Athletic_Revenue_M'] = df_final_ml['School_Athletic_Revenue_M'].fillna(df_final_ml['School_Athletic_Revenue_M'].mean())
df_final_ml['School_Total_Sport_Followers_K'] = df_final_ml['School_Total_Sport_Followers_K'].fillna(df_final_ml['School_Total_Sport_Followers_K'].mean())

print("--- MERGED DATA MATRIX SUCCESS ---")
print(df_final_ml[['Player', 'School', 'Position', 'School_Athletic_Revenue_M', 'NIL_Valuation']].head(10))
