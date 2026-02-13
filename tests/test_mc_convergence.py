from src.engine import PricingEngine
import numpy as np

def test_mc_convergence():
    engine = PricingEngine()
    S = 100
    K = 100
    T = 1
    sigma = 0.2

    bs = engine.price_bs(S,K,T,sigma,'call')

    paths = engine.simulate_paths(S,T,sigma)
    payoff = np.maximum(paths[-1] - K, 0)
    mc = np.exp(-engine.r*T) * payoff.mean()

    assert abs(bs - mc) < 5
