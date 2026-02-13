import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.local_vol import LocalVolatility

# Load synthetic surface
surface = pd.read_csv('data/surface_sample.csv')

S0 = 23000

lv = LocalVolatility()
call_surface = lv.build_call_surface(surface, S0)

K_grid, T_grid, local_vol = lv.compute_local_vol(call_surface)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(K_grid, T_grid, local_vol, cmap='plasma')

ax.set_title("Local Volatility Surface (Dupire)")
ax.set_xlabel("Strike")
ax.set_ylabel("Time to Maturity")
ax.set_zlabel("Local Volatility")

plt.tight_layout()
plt.savefig("local_vol_surface.png", dpi=300)
plt.show()

print("? Local volatility surface generated.")
