import numpy as np


def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class RedFeedforward:
    """Perceptrón multicapa (feedforward) entrenado con retropropagación y mini-lotes.

    Parámetros
    ----------
    capas_ocultas : tuple   Neuronas por capa oculta, p. ej. (8, 4).
    eta : float             Tasa de aprendizaje.
    epocas : int            Número de épocas.
    lote : int              Tamaño del mini-lote.
    semilla : int           Semilla aleatoria.
    """

    def __init__(self, capas_ocultas=(8,), eta=0.1, epocas=500, lote=16, semilla=None):
        self.capas_ocultas, self.eta, self.epocas, self.lote, self.semilla = capas_ocultas, eta, epocas, lote, semilla

    def _adelante(self, X):
        A = [X]
        for l, (W, b) in enumerate(zip(self.W_, self.b_)):
            z = A[-1] @ W + b
            A.append(_softmax(z) if l == len(self.W_) - 1 else _sigmoide(z))
        return A

    def fit(self, X, y, verbose=False):
        X = np.asarray(X, float); y = np.asarray(y).ravel()
        self.clases_ = np.unique(y)
        Y = (y[:, None] == self.clases_[None, :]).astype(float)
        rng = np.random.default_rng(self.semilla)
        tam = [X.shape[1], *self.capas_ocultas, len(self.clases_)]
        self.W_ = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(tam[:-1], tam[1:])]
        self.b_ = [np.zeros(b) for b in tam[1:]]
        self.perdida_ = []
        for ep in range(1, self.epocas + 1):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), self.lote):
                j = idx[i:i + self.lote]
                A = self._adelante(X[j])
                delta = (A[-1] - Y[j]) / len(j)
                for l in reversed(range(len(self.W_))):
                    gW, gb = A[l].T @ delta, delta.sum(axis=0)
                    if l > 0:
                        delta = (delta @ self.W_[l].T) * A[l] * (1 - A[l])
                    self.W_[l] -= self.eta * gW
                    self.b_[l] -= self.eta * gb
            P = self._adelante(X)[-1]
            self.perdida_.append(float(-np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))))
            if verbose and (ep == 1 or ep % max(1, self.epocas // 10) == 0):
                print(f"Época {ep:4d} | pérdida = {self.perdida_[-1]:.4f}")
        return self

    def predict_proba(self, X):
        return self._adelante(np.asarray(X, float))[-1]

    def predict(self, X):
        return self.clases_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
