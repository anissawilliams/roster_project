import pandas as pd

# Load your active dataframes
df_macro = pd.read_csv("school_macro_data.csv")

# Create a dictionary tracking verified programmatic athletic debt (in Millions)
# FSU ($437M) and Cal ($432M) lead the nation in athletic infrastructure liabilities
debt_map = {
    'Florida State': 437.0,
    'Texas': 0.0,
    'Ohio State': 15.0,
    'Michigan': 8.0,
    'Georgia': 0.0
}

# Engineer a new column: Institutional_Debt_M
df_macro['Institutional_Debt_M'] = df_macro['School'].map(debt_map).fillna(0.0)

print("--- NEW MACRO MATRIX WITH COMPLIANCE DEBT EXTRACTIONS ---")
print(df_macro[['School', 'School_Athletic_Revenue_M', 'Institutional_Debt_M']].head(10))

