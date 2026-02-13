import numpy as np
from scipy.optimize import brentq
from .black_scholes import black_scholes_price

def implied_volatility(price, S, K, T, r, option_type):
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0)
    except:
        return np.nan
