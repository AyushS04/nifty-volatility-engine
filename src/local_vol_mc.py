import numpy as np
from scipy.interpolate import RegularGridInterpolator

class LocalVolMonteCarlo:

    def __init__(self, K_grid, T_grid, local_vol_surface, r=0.06):
        self.r = r

        strikes = K_grid[0]
        maturities = T_grid[:,0]

        mean_vol = float(np.mean(local_vol_surface))

        self.vol_interp = RegularGridInterpolator(
            (maturities, strikes),
            local_vol_surface,
            bounds_error=False,
            fill_value=mean_vol   # ? STABLE fallback
        )

    def simulate(self, S0, T, steps=100, paths=2000):

        dt = T / steps
        times = np.linspace(0, T, steps+1)

        S = np.zeros((steps+1, paths))
        S[0] = S0

        for t in range(1, steps+1):

            Z = np.random.standard_normal(paths)

            vols = self.vol_interp(
                np.column_stack([
                    np.full(paths, times[t]),
                    S[t-1]
                ])
            )

            vols = np.maximum(vols, 0.05)  # safety floor

            S[t] = S[t-1] * np.exp(
                (self.r - 0.5*vols**2)*dt + vols*np.sqrt(dt)*Z
            )

        return S
