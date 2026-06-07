import pandas as pd
import numpy as np
import io

# 1. LOAD YOUR LIVE CSV STRUCTURES
# (Replace the string IO streams with your actual file paths)
roster_data = """roster id,firstName,lastName,team,weight,height,jersey,year,position,homeCity,homeState,homeCountry,homeLatitude,homeLongitude,homeCountyFIPS,recruitIds
1,Thomas,Castellanos,Florida State,195,70,1,3,QB,Waycross,GA,USA,0,0,0,101
2,Duce,Robinson,Florida State,225,78,8,2,WR,Phoenix,AZ,USA,0,0,0,102
3,Xavier,Chaplin,Florida State,315,79,70,3,OT,Seabrook,SC,USA,0,0,0,103
4,Caziah,Holmes,Florida State,210,71,26,4,RB,Cocoa,FL,USA,0,0,0,104"""

transfer_data = """season,firstName,lastName,position,origin,destination,transferDate,rating,stars,eligibility,direction
2025,Thomas,Castellanos,QB,Boston College,Florida State,2024-12-15,0.92,4.0,Jr,Inbound
2025,Duce,Robinson,WR,USC,Florida State,2025-01-10,0.98,5.0,So,Inbound"""

val_data = """name,position,nil_value,nil_source,social_followers,valuation_date
Duce Robinson,WR,1200000,On3,95000,2025-06-01
Mandrell Desir,DL,810000,On3,42000,2025-06-01
Xavier Chaplin,OT,743000,On3,38000,2025-06-01
Micahi Danzy,WR,386000,On3,28000,2025-06-01
Thomas Castellanos,QB,375000,On3,52000,2025-06-01
Caziah Holmes,RB,95000,Estimated,14000,2025-06-01"""

# df_roster = pd.read_csv(io.StringIO(roster_data))
# df_transfers = pd.read_csv(io.StringIO(transfer_data))
# df_vals = pd.read_csv(io.StringIO(val_data))
df_roster = pd.read_csv("../../data/fsu/fsu_roster_2025.csv")
df_transfers = pd.read_csv("../../data/fsu/fsu_transfers_2025.csv")
df_vals = pd.read_csv("../../.archive/fsu_nil_valuations.csv")

# ==========================================
# 2. STRING ALIGNMENT ENGINE
# ==========================================
# Build matched lookup strings across your files
df_roster['Full_Name'] = df_roster['firstName'].str.strip() + " " + df_roster['lastName'].str.strip()
df_transfers['Full_Name'] = df_transfers['firstName'].str.strip() + " " + df_transfers['lastName'].str.strip()

# Create a clean, isolated transfer lookup table
df_trans_lookup = df_transfers[['Full_Name', 'stars', 'rating']].copy()
df_trans_lookup['Is_Transfer'] = 1.0

# ==========================================
# 3. INTER-TABULAR RELATIONAL JOIN
# ==========================================
# Merge the master roster with transfer pedigree and individual clout metrics
df_merged = pd.merge(df_roster, df_trans_lookup, on="Full_Name", how="left")
df_merged['Is_Transfer'] = df_merged['Is_Transfer'].fillna(0.0)

# Fill missing recruit ratings with baseline averages to protect model arrays
df_merged['stars'] = df_merged['stars'].fillna(3.0)
df_merged['rating'] = df_merged['rating'].fillna(0.85)

# Final step: Merge the player individual valuation target list
df_master_ml = pd.merge(df_merged, df_vals[['name', 'nil_value', 'social_followers']], left_on="Full_Name", right_on="name", how="inner")

print("--- UNIFIED INDIVIDUAL ML DATA MATRIX ---")
print(df_master_ml[['Full_Name', 'position', 'stars', 'social_followers', 'nil_value']])
print(f"Final ML Matrix Dimension: {df_master_ml.shape} (Rows, Highly-Dense Platforms)")
print(df_master_ml.head(5))
df_master_ml.to_csv("data/master_ml_data.csv", index=False)
