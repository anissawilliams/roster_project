import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

r = requests.get(
    "https://api.collegefootballdata.com/player/portal",
    headers=headers,
    params={"year": 2025}
)

df = pd.DataFrame(r.json())
print(f"Total portal entries: {len(df)}")

# filter to FSU
fsu_in  = df[df["destination"] == "Florida State"]
fsu_out = df[df["origin"] == "Florida State"]

print(f"Incoming to FSU: {len(fsu_in)}")
print(f"Outgoing from FSU: {len(fsu_out)}")

print("\nIncoming sample:")
print(fsu_in.head(3).to_string())

print("\nOutgoing sample:")
print(fsu_out.head(3).to_string())

# save both
fsu_in["direction"]  = "Incoming"
fsu_out["direction"] = "Outgoing"
transfers = pd.concat([fsu_in, fsu_out])
transfers.to_csv("fsu_transfers_2025.csv", index=False)
print(f"\nSaved {len(transfers)} FSU transfer entries")