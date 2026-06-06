import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}"}
params = {"team": "Florida State", "year": 2025}

r = requests.get(
    "https://api.collegefootballdata.com/roster",
    headers=headers,
    params=params
)

print(r.status_code)
data = r.json()
print(data[:2])  # peek at first 2 records

df = pd.DataFrame(data)
df.to_csv("fsu_roster_2025.csv", index=False)
print(f"Saved {len(df)} players")
print(df.columns.tolist())