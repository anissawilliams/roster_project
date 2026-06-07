import pandas as pd

# 1. Load your newly discovered Excel file
# (Replace "social_data.xlsx" with your actual file path)
df_excel = pd.read_excel("data/Teams_Social_Media.xlsx")

# 2. FILTERING STEP: Isolate exactly what your model needs
# Filter out non-football sports and keep only the sports-specific social accounts
df_filtered = df_excel[
    (df_excel['Account'].str.lower() == 'football')
]

# 3. CONVERT THE DYNAMIC ROWS TO WIDE FEATURE COLUMNS
df_filtered['Feature_Name'] = df_filtered['Platform'].str.upper() + "_" + df_filtered['Account'].str.replace(' ', '_').str.upper()

df_pivoted_social = df_filtered.pivot_table(
    index=['School'],
    columns='Feature_Name',
    values='Followers',
    aggfunc='max'
).fillna(0).reset_index()

# 4. THE STRING MAPPING CODE ENGINE
# This bridges official university names to the compact strings in our player file
school_name_mapping = {
    'Auburn University': 'Auburn',
    'University of Texas at Austin': 'Texas',
    'Ohio State University': 'Ohio State',
    'Louisiana State University': 'LSU',
    'Texas Tech University': 'Texas Tech',
    'University of Michigan': 'Michigan',
    'University of South Carolina': 'South Carolina',
    'Oklahoma State University': 'Oklahoma State',
    'University of Oregon': 'Oregon',
    'University of Alabama': 'Alabama',
    'University of Colorado Boulder': 'Colorado',
    'University of Notre Dame': 'Notre Dame',
    'University of Miami': 'Miami',
    'University of Georgia': 'Georgia',
    'University of Mississippi': 'Ole Miss',
    'Pennsylvania State University': 'Penn State',
    'University of Southern California': 'USC',
    'Florida State University': 'Florida State'
}

# Apply mapping and prune unneeded schools that don't match our student list
df_pivoted_social['Mapped_School'] = df_pivoted_social['School'].map(school_name_mapping)
df_social_final = df_pivoted_social.dropna(subset=['Mapped_School']).drop(columns=['School'])

# 5. INNER JOIN: Merge the student players directly with the wide social features
df_players = pd.read_csv("../../data/training/nil_data.csv")

df_final_training_matrix = pd.merge(
    df_players,
    df_social_final,
    left_on="School",
    right_on="Mapped_School",
    how="inner" # Drops any school from the excel sheet you don't care about tracking
).drop(columns=['Mapped_School'])

print("--- PIPELINE INTEGRATION SUCCESS ---")
print(f"Final ML Matrix Dimension: {df_final_training_matrix.shape} (Rows, Highly-Dense Platforms)")
print(df_final_training_matrix.head(5))
df_final_training_matrix.to_csv("data/final_training_matrix.csv", index=False)
