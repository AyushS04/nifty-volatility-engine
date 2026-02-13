import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.local_vol import LocalVolatility
from src.local_vol_mc import LocalVolMonteCarlo
from src.monte_carlo import monte_carlo_paths

# Load synthetic surface
surface = pd.read_csv('data/surface_sample.csv')

S0 = 23000
T = 0.5

lv = LocalVolatility()
K_grid, T_grid, local_vol = lv.compute_local_vol(surface, S0)

# Local Vol MC
lv_mc = LocalVolMonteCarlo(K_grid, T_grid, local_vol)
paths_lv = lv_mc.simulate(S0, T, steps=100, paths=2000)

# GBM MC
sigma_flat = surface['implied_vol'].mean()
paths_gbm = monte_carlo_paths(S0, T, 0.06, sigma_flat, steps=100, paths=2000)

# Compare terminal distributions
plt.figure()

plt.hist(paths_gbm[-1], bins=50, alpha=0.5, density=True, label='GBM')
plt.hist(paths_lv[-1], bins=50, alpha=0.5, density=True, label='Local Vol')

plt.title('Terminal Distribution: GBM vs Local Vol')
plt.legend()
plt.tight_layout()
plt.savefig("lv_vs_gbm.png", dpi=300)
plt.show()

print("? Comparison plot saved.")
