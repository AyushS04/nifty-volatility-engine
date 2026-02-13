import numpy as np

class CubicSmile:

    def fit(self, moneyness, iv, weights=None):
        if weights is None:
            weights = np.ones_like(iv)

        X = np.vstack([
            np.ones_like(moneyness),
            moneyness,
            moneyness**2,
            moneyness**3
        ]).T

        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ iv
        self.params = beta
        return beta

    def predict(self, moneyness):
        a,b,c,d = self.params
        return a + b*moneyness + c*moneyness**2 + d*moneyness**3
