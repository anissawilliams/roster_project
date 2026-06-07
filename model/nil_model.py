# NIL valuation model for RosterEdge
# change the three variables below to run for a different team
# using gradient boosting + depth chart role adjustment
# training data is 25 players from On3/public sources

import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---- change these to run for a different team ----
YEAR_TARGET = 2025
TARGET_TEAM = 'Florida State'
TEAM_SLUG   = 'fsu'
# --------------------------------------------------

TEAM_CONFIG = {
    'fsu': {
        'roster':  f'data/fsu/fsu_roster_processed_{YEAR_TARGET}.csv',
        'depth':   f'data/fsu/fsu_depth_chart_{YEAR_TARGET}.csv',
        'on3':     f'data/fsu/fsu_nil_on3_raw.csv',
        'output':  f'data/fsu/fsu_nil_valuations_final.csv',
        # roster col that has player name - processed fsu file already has Full_Name
        'name_col': 'Full_Name',
        'build_name': False,
        # name fixes for depth chart mismatches
        'name_fixes': {
            'Amare Williams': 'Amaree Williams',
            'LeWayne McCoy':  'Lawayne McCoy',
        },
        # individually verified instagram follower counts
        'known_social': {
            'Squirrel White':    30700,
            'Tommy Castellanos': 107000,
            'Roydell Williams':  20500,
            'Jaylen King':       2285,
            'Jaylin Lucas':      10200,
            'Earl Little Jr.':   42800,
            'Markeston Douglas': 4675,
        },
    },
    'uga': {
        'roster':  f'data/uga/uga_roster_{YEAR_TARGET}.csv',
        'depth':   f'data/uga/uga_depth_chart_{YEAR_TARGET}.csv',
        'on3':     f'data/uga/uga_nil_on3_raw.csv',
        'output':  f'data/uga/uga_nil_valuations_final.csv',
        'name_col': 'Full_Name',
        'build_name': True,  # needs firstName + lastName combined
        'name_fixes': {
            'Frazier':    'Nate Frazier',
            'Kj Bolden':  'KJ Bolden',
        },
        'known_social': {},  # no individual lookups yet for uga
    },
}

if TEAM_SLUG not in TEAM_CONFIG:
    print(f"Unknown team slug '{TEAM_SLUG}'. Add it to TEAM_CONFIG first.")
    sys.exit(1)

cfg = TEAM_CONFIG[TEAM_SLUG]

# load data
df_train = pd.read_csv("data/training/final_training_matrix.csv")
df_roster = pd.read_csv(cfg['roster'])
df_depth  = pd.read_csv(cfg['depth'])
df_nil    = pd.read_csv(cfg['on3'])

# build full name if needed
if cfg['build_name']:
    df_roster['Full_Name'] = df_roster['firstName'].str.strip() + ' ' + df_roster['lastName'].str.strip()

print(f"{TARGET_TEAM} roster: {len(df_roster)} players")

# normalize positions
pos_map = {
    'OT': 'OL', 'IOL': 'OL', 'CB': 'DB', 'S': 'DB',
    'LB': 'DL', 'LS': 'OL', 'P':  'OL', 'PK': 'OL',
    'DT': 'DL', 'DE': 'DL',
}
df_train['Position']  = df_train['Position'].replace(pos_map)
df_roster['position'] = df_roster['position'].replace(pos_map)

# feature engineering
social_cols = ['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL',
               'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']

df_train['total_social'] = df_train[social_cols].sum(axis=1)
df_train['log_social']   = np.log1p(df_train['total_social'])

pos_mult = {
    'QB': 2.4, 'WR': 1.5, 'RB': 0.9, 'DL': 1.1,
    'DB': 0.8, 'OL': 0.7, 'TE': 0.7
}
df_train['pos_weight']   = df_train['Position'].map(pos_mult).fillna(0.8)
df_train['rank_score']   = 1.0 - (df_train['Rank'] / df_train['Rank'].max())
df_train['social_x_pos'] = df_train['log_social'] * df_train['pos_weight']

features = ['log_social', 'pos_weight', 'rank_score', 'social_x_pos']
X_train  = df_train[features]
y_train  = df_train['NIL_Valuation'].values
y_log    = np.log10(y_train)

# LOOCV - leave one out because n=25 is small
loo = LeaveOneOut()
oof_preds = np.zeros(len(df_train))
for train_idx, test_idx in loo.split(X_train):
    m = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
    m.fit(X_train.iloc[train_idx], y_log[train_idx])
    oof_preds[test_idx] = m.predict(X_train.iloc[test_idx])

loocv_mae = mean_absolute_error(y_train, 10**oof_preds)
loocv_r2  = r2_score(y_train, 10**oof_preds)
print(f"LOOCV MAE: ${loocv_mae:,.0f} | R2: {loocv_r2:.4f}")

# train final model
model = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
model.fit(X_train, y_log)

# depth chart role lookup
role_mult = {'starter': 1.45, 'backup': 1.15, 'depth': 1.0}

def fix_name(raw, fixes):
    raw = str(raw).strip()
    if raw in fixes:
        return fixes[raw]
    # title case handles all-caps depth charts
    return raw.title()

role_lookup = {}
for _, row in df_depth.iterrows():
    for col, role in [('Starter', 'starter'), ('Backup', 'backup'),
                      ('Depth', 'depth'), ('Reserve', 'depth')]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip() and str(val).strip() != '—':
            name = fix_name(val, cfg['name_fixes'])
            role_lookup[name] = role

print(f"Depth chart entries mapped: {len(role_lookup)}")

# social follower imputation by position + year
social_impute = {
    ('QB', 4): 28000, ('QB', 3): 18000, ('QB', 2): 9000,  ('QB', 1): 5000,
    ('WR', 4): 22000, ('WR', 3): 16000, ('WR', 2): 8000,  ('WR', 1): 4500,
    ('RB', 4): 18000, ('RB', 3): 14000, ('RB', 2): 7000,  ('RB', 1): 4000,
    ('DL', 4): 14000, ('DL', 3): 10000, ('DL', 2): 6000,  ('DL', 1): 3500,
    ('DB', 4): 12000, ('DB', 3): 9000,  ('DB', 2): 5500,  ('DB', 1): 3000,
    ('OL', 4): 10000, ('OL', 3): 7500,  ('OL', 2): 4500,  ('OL', 1): 2500,
    ('TE', 4): 11000, ('TE', 3): 8000,  ('TE', 2): 5000,  ('TE', 1): 3000,
}
rank_impute = {
    ('QB', 4): 0.30, ('QB', 3): 0.20, ('QB', 2): 0.10, ('QB', 1): 0.05,
    ('WR', 4): 0.28, ('WR', 3): 0.18, ('WR', 2): 0.08, ('WR', 1): 0.04,
    ('RB', 4): 0.25, ('RB', 3): 0.16, ('RB', 2): 0.07, ('RB', 1): 0.03,
    ('DL', 4): 0.22, ('DL', 3): 0.14, ('DL', 2): 0.06, ('DL', 1): 0.02,
    ('DB', 4): 0.20, ('DB', 3): 0.12, ('DB', 2): 0.05, ('DB', 1): 0.02,
    ('OL', 4): 0.18, ('OL', 3): 0.10, ('OL', 2): 0.04, ('OL', 1): 0.01,
    ('TE', 4): 0.18, ('TE', 3): 0.10, ('TE', 2): 0.04, ('TE', 1): 0.01,
}

# known nil floors from on3 file
known_nil = {}
for _, row in df_nil.iterrows():
    nil_val = row['nil_value'] if pd.notna(row.get('nil_value')) else row.get('whisper_value')
    if pd.notna(nil_val):
        known_nil[row['Full_Name']] = nil_val

# merge team-level known social with any verified individual counts
known_social = dict(zip(
    df_nil['Full_Name'],
    df_nil.get('social_followers', pd.Series([None]*len(df_nil)))
))
known_social = {k: v for k, v in known_social.items() if pd.notna(v)}
known_social.update(cfg['known_social'])

# build inference rows
rows = []
for _, row in df_roster.iterrows():
    name = row['Full_Name']
    pos  = row['position']
    year = int(row['year']) if not pd.isna(row['year']) else 1
    yb   = min(year, 4)

    social = known_social.get(name, social_impute.get((pos, yb), 3000))
    source = 'verified' if name in known_social else 'imputed'

    rows.append({
        'Full_Name':        name,
        'position':         pos,
        'year':             year,
        'social_followers': social,
        'social_source':    source,
        'rank_score':       rank_impute.get((pos, yb), 0.02),
        'depth_role':       role_lookup.get(name, 'depth'),
    })

df_team = pd.DataFrame(rows)
df_team['log_social']    = np.log1p(df_team['social_followers'])
df_team['pos_weight']    = df_team['position'].map(pos_mult).fillna(0.8)
df_team['social_x_pos']  = df_team['log_social'] * df_team['pos_weight']

X_team    = df_team[features]
preds_log = model.predict(X_team)
df_team['base_nil'] = np.round(10 ** preds_log, -2)

# apply role multiplier
df_team['role_mult']    = df_team['depth_role'].map(role_mult)
df_team['predicted_nil'] = np.round(df_team['base_nil'] * df_team['role_mult'], -2)

# on3/whisper as floor
for name, nil_val in known_nil.items():
    mask = df_team['Full_Name'] == name
    if not mask.any():
        # try first + last name partial match for suffix mismatches (e.g. Greene vs Greene III)
        parts = name.split()
        if len(parts) >= 2:
            mask = df_team['Full_Name'].str.contains(parts[0] + ' ' + parts[1], na=False)
    if mask.any():
        curr = df_team.loc[mask, 'predicted_nil'].values[0]
        df_team.loc[mask, 'predicted_nil'] = max(curr, nil_val)
        df_team.loc[mask, 'social_source'] = 'verified'

df_team['confidence'] = df_team['social_source'].map({'verified': 'high', 'imputed': 'estimated'})

df_out = df_team[['Full_Name', 'position', 'year', 'social_followers', 'social_source',
                   'depth_role', 'role_mult', 'base_nil', 'predicted_nil', 'confidence']].copy()
df_out = df_out.sort_values('predicted_nil', ascending=False).reset_index(drop=True)
df_out['predicted_nil_fmt'] = df_out['predicted_nil'].map(lambda x: f"${x:,.0f}")

df_out.to_csv(cfg['output'], index=False)

print(df_out[['Full_Name', 'position', 'depth_role', 'predicted_nil_fmt', 'confidence']].head(20).to_string(index=False))
print(f"\nsaved to {cfg['output']}")