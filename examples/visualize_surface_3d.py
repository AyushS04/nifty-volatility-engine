import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

surface = pd.read_csv("data/surface_sample.csv", parse_dates=['date','expiry'])

latest_date = surface['date'].max()
surface = surface[surface['date'] == latest_date]

surface = surface.dropna(subset=['implied_vol'])

X = surface['strike']
Y = surface['time_to_maturity']
Z = surface['implied_vol']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title(f"Implied Volatility Surface ({latest_date.date()})")
ax.set_xlabel("Strike")
ax.set_ylabel("Time to Maturity")
ax.set_zlabel("Implied Volatility")

plt.tight_layout()
plt.show()
