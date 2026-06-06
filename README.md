# RosterEdge – Prototype v0.1

A scaled-down roster viewer for college football programs, scoped to Florida State for prototype demonstration.

## What it does
- Loads FSU 2026 roster data from a local CSV
- Displays a filterable roster table (by position, class year, name search)
- Shows position breakdown and class year charts
- Summary metrics (total players, positions, states represented)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
rosteredge-prototype/

├── app.py                 # Streamlit app
├── data 
├──     fsu_roster_2026.csv    # FSU roster data
├── requirements.txt
└── README.md
```

## Next steps (Intermediate milestone)
- Connect to a SQLite/PostgreSQL database instead of CSV
- Add multi-team support
- Begin NIL valuation data integration
