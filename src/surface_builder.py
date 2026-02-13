import pandas as pd
import numpy as np
from .implied_vol import implied_volatility

class VolatilitySurface:

    def __init__(self, r=0.06):
        self.r = r

    def build_surface(self, options_df, spot_df):
        rows = []

        for _, row in options_df.iterrows():
            S_row = spot_df[spot_df['date'] == row['date']]
            if S_row.empty:
                continue

            S = S_row['close'].values[0]
            T = (row['expiry'] - row['date']).days / 365.0
            if T <= 0:
                continue

            iv = implied_volatility(
                row['close'], S, row['strike'], T, self.r, row['option_type']
            )

            rows.append({
                "date": row['date'],
                "expiry": row['expiry'],
                "strike": row['strike'],
                "implied_vol": iv,
                "time_to_maturity": T
            })

        return pd.DataFrame(rows)
