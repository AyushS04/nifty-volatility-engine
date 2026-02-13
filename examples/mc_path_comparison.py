import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.local_vol import LocalVolatility
from src.local_vol_mc import LocalVolMonteCarlo
from src.monte_carlo import monte_carlo_paths

# Load surface
surface = pd.read_csv('data/surface_sample.csv')

S0 = 23000
T = 0.5
steps = 180
paths = 300

# Build Local Vol
lv = LocalVolatility()
K_grid, T_grid, local_vol = lv.compute_local_vol(surface, S0)
lv_mc = LocalVolMonteCarlo(K_grid, T_grid, local_vol)

# Simulate
paths_lv = lv_mc.simulate(S0, T, steps=steps, paths=paths)
sigma_flat = surface['implied_vol'].mean()
paths_gbm = monte_carlo_paths(S0, T, 0.06, sigma_flat, steps=steps, paths=paths)

time_axis = np.linspace(0, T, steps+1)

plt.figure(figsize=(10,6))

# Plot GBM
for i in range(paths):
    plt.plot(time_axis, paths_gbm[:,i], color='blue', alpha=0.05)

# Plot Local Vol
for i in range(paths):
    plt.plot(time_axis, paths_lv[:,i], color='orange', alpha=0.05)

plt.title("Monte Carlo Path Evolution: GBM vs Local Vol")
plt.xlabel("Time")
plt.ylabel("Index Level")
plt.tight_layout()
plt.savefig("mc_path_comparison.png", dpi=300)
plt.show()

print("? Monte Carlo path comparison saved.")
