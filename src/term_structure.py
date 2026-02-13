import pandas as pd

def build_atm_term_structure(surface_df, spot_df):

    atm_rows = []

    for (date, expiry), group in surface_df.groupby(['date','expiry']):

        S_row = spot_df[spot_df['date'] == date]
        if S_row.empty:
            continue

        S = S_row['close'].values[0]

        group = group.copy()
        group['distance'] = abs(group['strike'] - S)

        atm = group.sort_values('distance').iloc[0]

        atm_rows.append({
            "date": date,
            "expiry": expiry,
            "time_to_maturity": atm['time_to_maturity'],
            "atm_iv": atm['implied_vol']
        })

    return pd.DataFrame(atm_rows)
