# RosterEdge – Prototype v0.1

A roster intelligence viewer for college football programs, scoped to Florida State for prototype demonstration. Combines real API data with NIL valuations into a single Streamlit interface.

## What it does

- **Roster tab** — 108 real FSU 2025 players via CFBD API, filterable by position, class year, and name
- **NIL Valuations tab** — On3-sourced valuations for top players, estimated for the rest, with position breakdown chart
- **Transfer Portal tab** — Real 2025 portal data (22 incoming, 34 outgoing) with ratings, eligibility, and positional gap analysis

## Project structure

```
roster_project/
├── data/
│   ├── ingestion/
│   │   ├── fetch_roster_data.py       # pulls FSU roster from CFBD API
│   │   └── fetch_transfer_data.py     # pulls FSU portal data from CFBD API
│   ├── fsu_nil_valuations.csv         # On3 + estimated NIL values
│   ├── fsu_roster_2025.csv            # real CFBD roster data
│   ├── fsu_roster_2026.csv            # placeholder for next season
│   ├── fsu_transfers.csv              # legacy/synthetic transfer data
│   └── fsu_transfers_2025.csv         # real CFBD portal data
├── .env                               # local only — never committed
├── .gitignore
├── app.py                             # Streamlit app
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fetch fresh data

```bash
python data/ingestion/fetch_roster_data.py
python data/ingestion/fetch_transfer_data.py
```

Requires a `.env` file with:
```
CFBD_API_KEY=your_key_here
```

Get a free key at [collegefootballdata.com](https://collegefootballdata.com/key).

## Data sources

| Dataset | Source |
|---|---|
| Roster | CFBD API `/roster` endpoint |
| Transfer portal | CFBD API `/player/portal` endpoint |
| NIL valuations | On3 NIL Index (top players) + estimated |

## Next steps (Intermediate milestone)
- Load all three datasets into SQLite instead of CSV files
- Add multi-team support
- Begin ML modeling for NIL valuation estimation