from .black_scholes import black_scholes_price
from .monte_carlo import monte_carlo_paths
import numpy as np

class PricingEngine:

    def __init__(self, r=0.06):
        self.r = r

    def price_bs(self, S, K, T, sigma, option_type):
        return black_scholes_price(S, K, T, self.r, sigma, option_type)

    def simulate_paths(self, S, T, sigma):
        return monte_carlo_paths(S, T, self.r, sigma)
