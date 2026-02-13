from src.engine import PricingEngine
import matplotlib.pyplot as plt

engine = PricingEngine()

S0 = 23000
T = 0.25
sigma = 0.15

paths = engine.simulate_paths(S0, T, sigma)

plt.plot(paths[:, :50])
plt.title("Monte Carlo Simulation")
plt.show()
