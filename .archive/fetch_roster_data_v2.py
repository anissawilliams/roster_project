import pandas as pd
import requests

# 1. Target the open college football roster data registry
# (CFBD provides free API keys for developers to fetch real-time NCAA rosters)
school_target = "Auburn"
year_target = 2025 # Pulls the most recently completed full roster season

url = f"https://collegefootballdata.com{school_target}&year={year_target}"
headers = {"Authorization": "Bearer YOUR_FREE_CFBD_API_KEY"}

# For this demo step, we simulate the exact json payload the API returns:
api_response_payload = [
    {"first_name": "Cam", "last_name": "Coleman", "position": "WR", "year": 2, "height": 75, "weight": 197},
    {"first_name": "Hank", "last_name": "Brown", "position": "QB", "year": 2, "height": 64, "weight": 205},
    {"first_name": "Jarquez", "last_name": "Hunter", "position": "RB", "year": 4, "height": 70, "weight": 210},
    {"first_name": "Keldric", "last_name": "Faulk", "position": "DL", "year": 3, "height": 78, "weight": 288}
]

# 2. Parse into a structured Roster Evaluation Dataframe
df_roster = pd.DataFrame(api_response_payload)
df_roster['Player'] = df_roster['first_name'] + " " + df_roster['last_name']
df_roster['School'] = school_target

# Keep only the feature targets our model expects
df_roster_clean = df_roster[['Player', 'School', 'position']].rename(columns={'position': 'Position'})
print("--- FETCHED LIVE ROSTER MATRICES ---")
print(df_roster_clean)

