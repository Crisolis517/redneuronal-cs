import numpy as np


class Perceptron:
    """Perceptrón simple de Rosenblatt con función escalón.

    Parámetros
    ----------
    eta : float      Tasa de aprendizaje.
    epocas : int     Número máximo de épocas.
    semilla : int    Semilla para inicializar los pesos.
    """

    def __init__(self, eta=0.1, epocas=50, semilla=None):
        self.eta, self.epocas, self.semilla = eta, epocas, semilla

    def fit(self, X, y, verbose=False):
        X, y = np.asarray(X, float), np.asarray(y).ravel()
        rng = np.random.default_rng(self.semilla)
        self.w_ = rng.uniform(-0.5, 0.5, X.shape[1])
        self.b_ = rng.uniform(-0.5, 0.5)
        self.errores_ = []
        for ep in range(1, self.epocas + 1):
            err = 0
            for xi, yi in zip(X, y):
                e = yi - self.predict(xi[None, :])[0]
                self.w_ += self.eta * e * xi
                self.b_ += self.eta * e
                err += int(e != 0)
            self.errores_.append(err)
            if verbose:
                print(f"Época {ep:3d} | errores = {err}")
            if err == 0:
                break
        return self

    def net_input(self, X):
        return np.asarray(X, float) @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
