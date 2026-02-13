import numpy as np

def monte_carlo_paths(S0, T, r, sigma, steps=100, paths=1000):
    dt = T / steps
    S = np.zeros((steps+1, paths))
    S[0] = S0

    for t in range(1, steps+1):
        Z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    return S
