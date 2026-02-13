import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from .black_scholes import black_scholes_price

class LocalVolatility:

    def __init__(self, r=0.06):
        self.r = r

    def smooth_iv_surface(self, surface_df):

        K = surface_df['strike'].values
        T = surface_df['time_to_maturity'].values
        iv = surface_df['implied_vol'].values

        X = np.column_stack([
            np.ones_like(K),
            K,
            T,
            K**2,
            T**2,
            K*T
        ])

        beta, *_ = np.linalg.lstsq(X, iv, rcond=None)

        iv_smooth = (
            beta[0]
            + beta[1]*K
            + beta[2]*T
            + beta[3]*(K**2)
            + beta[4]*(T**2)
            + beta[5]*(K*T)
        )

        surface_df = surface_df.copy()
        surface_df['iv_smooth'] = iv_smooth

        return surface_df

    def compute_local_vol(self, surface_df, S0):

        surface_df = self.smooth_iv_surface(surface_df)

        rows = []

        for _, row in surface_df.iterrows():

            C = black_scholes_price(
                S0,
                row['strike'],
                row['time_to_maturity'],
                self.r,
                row['iv_smooth'],
                'call'
            )

            rows.append({
                "strike": row['strike'],
                "T": row['time_to_maturity'],
                "call_price": C
            })

        call_surface = pd.DataFrame(rows)

        strikes = np.sort(call_surface['strike'].unique())
        maturities = np.sort(call_surface['T'].unique())

        K_grid, T_grid = np.meshgrid(strikes, maturities)

        C_grid = griddata(
            (call_surface['strike'], call_surface['T']),
            call_surface['call_price'],
            (K_grid, T_grid),
            method='linear'
        )

        C_grid = np.nan_to_num(C_grid, nan=0.0)

        dC_dT = np.gradient(C_grid, maturities, axis=0)
        d2C_dK2 = np.gradient(
            np.gradient(C_grid, strikes, axis=1),
            strikes,
            axis=1
        )

        gamma_floor = 1e-6
        d2C_dK2 = np.where(np.abs(d2C_dK2) < gamma_floor, gamma_floor, d2C_dK2)

        local_var = dC_dT / (0.5 * (K_grid**2) * d2C_dK2)
        local_var = np.maximum(local_var, 0)

        local_vol = np.sqrt(local_var)

        return K_grid, T_grid, local_vol
