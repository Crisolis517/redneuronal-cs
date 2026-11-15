"""redneuronal_cs: redes neuronales simples programadas desde cero con NumPy.

Autor: Cristian Solís - Maestría en Estadística (Ciencia de Datos e IA), ESPOCH.
"""
from .perceptron import Perceptron
from .feedforward import RedFeedforward
from .utils import escalar_minmax, dividir_datos, matriz_confusion, exactitud

__version__ = "0.1.0"
__all__ = ["Perceptron", "RedFeedforward", "escalar_minmax", "dividir_datos", "matriz_confusion", "exactitud"]
