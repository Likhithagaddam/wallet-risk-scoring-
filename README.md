wallet_risk_scoring.py
----------------------------------

import pandas as pd

import requests

import time

from sklearn.preprocessing import MinMaxScaler

# 1. Load Wallets
df_wallets = pd.read_csv('wallets.csv')

wallets = df_wallets['wallet_id'].dropna().unique().tolist()

# 2. API Setup
API_KEY = 'cqt_rQhwk6Y3YyQtTdvgFWghBRD3fygd'  

CHAIN_ID = '1'

BASE_URL = 'https://api.covalenthq.com/v1'

wallet_features = {}

# 3. Fetch Compound Data

def fetch_compound_data(wallet):

    endpoint = f'{BASE_URL}/{CHAIN_ID}/address/{wallet}/stacks/compound_v2/'
    params = {'key': API_KEY}
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code != 200:
            print(f"[!] Failed API for wallet: {wallet}")
            return None
        return response.json().get('data', {}).get('items', [])
    except Exception as e:
        print(f"Error for {wallet}: {e}")
        return None

# 4. Extract Features
for wallet in wallets:

    print(f"Processing: {wallet}")
    data = fetch_compound_data(wallet)
    time.sleep(1.1)  # sleep to avoid rate limiting

    total_supply = 0
    total_borrow = 0
    num_assets = 0

    if data:
        for item in data:
            try:
                supplied = float(item.get('supply_balance_underlying', 0) or 0)
                borrowed = float(item.get('borrow_balance_underlying', 0) or 0)
            except:
                supplied = 0
                borrowed = 0

            total_supply += supplied
            total_borrow += borrowed

            if supplied > 0 or borrowed > 0:
                num_assets += 1

    else:
        # Debug: Tell if data is empty
        print(f"[!] No data for {wallet} — assigning 0s")

    net_position = total_supply - total_borrow

    wallet_features[wallet] = {
        'total_supply': total_supply,
        'total_borrow': total_borrow,
        'net_position': net_position,
        'num_assets': num_assets
    }

# 5. Create DataFrame
df = pd.DataFrame.from_dict(wallet_features, orient='index')

df.reset_index(inplace=True)

df.rename(columns={'index': 'wallet_id'}, inplace=True)

# Debug: Check values before normalization
print("\n[DEBUG] Raw Wallet Stats:\n", df.head())

# 6. Normalize & Score
cols_to_scale = ['total_borrow', 'total_supply', 'net_position', 'num_assets']

# Handle all-zero edge case
if df[cols_to_scale].nunique().max() == 1:
    print("\n[!] All values are same or zero — can't normalize. Adding noise for testing.")
    df[cols_to_scale] = df[cols_to_scale] + pd.Series(range(1, len(df)+1)).values.reshape(-1,1)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[cols_to_scale])
df_scaled = pd.DataFrame(df_scaled, columns=[col + '_norm' for col in cols_to_scale])

df = pd.concat([df, df_scaled], axis=1)

# Risk scoring formula
df['score'] = (
    df['total_borrow_norm'] * 0.4 +
    (1 - df['total_supply_norm']) * 0.2 +
    (1 - df['net_position_norm']) * 0.3 +
    (1 - df['num_assets_norm']) * 0.1
) * 1000

df['score'] = df['score'].round().astype(int)

# 7. Save Output
df_final = df[['wallet_id', 'score']]

df_final.to_csv('wallet_scores.csv', index=False)

print("\n Done! Scores saved to 'wallet_scores.csv'")
