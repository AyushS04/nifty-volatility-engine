import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from .black_scholes import black_scholes_price

class LocalVolatility:

    def __init__(self, r=0.06):
        self.r = r

    def build_call_surface(self, surface_df, S0):
        calls = []

        for _, row in surface_df.iterrows():
            C = black_scholes_price(
                S0,
                row['strike'],
                row['time_to_maturity'],
                self.r,
                row['implied_vol'],
                'call'
            )

            calls.append({
                "strike": row['strike'],
                "T": row['time_to_maturity'],
                "call_price": C
            })

        return pd.DataFrame(calls)

    def compute_local_vol(self, call_surface):

        strikes = np.sort(call_surface['strike'].unique())
        maturities = np.sort(call_surface['T'].unique())

        K_grid, T_grid = np.meshgrid(strikes, maturities)

        C_values = griddata(
            (call_surface['strike'], call_surface['T']),
            call_surface['call_price'],
            (K_grid, T_grid),
            method='cubic'
        )

        dC_dT = np.gradient(C_values, maturities, axis=0)
        d2C_dK2 = np.gradient(
            np.gradient(C_values, strikes, axis=1),
            strikes,
            axis=1
        )

        local_var = dC_dT / (0.5 * (K_grid**2) * d2C_dK2)

        local_var = np.maximum(local_var, 0)

        local_vol = np.sqrt(local_var)

        return K_grid, T_grid, local_vol
