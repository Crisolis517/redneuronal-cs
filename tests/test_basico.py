import numpy as np
from redneuronal_cs import Perceptron, RedFeedforward


def test_perceptron_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y = np.array([0, 0, 0, 1])
    assert Perceptron(eta=0.1, epocas=100, semilla=1).fit(X, y).score(X, y) == 1.0


def test_feedforward_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y = np.array([0, 1, 1, 0])
    red = RedFeedforward(capas_ocultas=(4,), eta=1.0, epocas=3000, lote=4, semilla=0).fit(X, y)
    assert red.score(X, y) == 1.0
