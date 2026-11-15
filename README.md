# redneuronal-cs

Paquete de Python con **redes neuronales simples programadas desde cero con NumPy**, desarrollado para la asignatura *Redes Neuronales y Deep Learning* de la Maestría en Estadística, mención Ciencia de Datos e Inteligencia Artificial (ESPOCH).

Autor: **Cristian Solís**

## Contenido

| Módulo | Descripción |
|---|---|
| `Perceptron` | Perceptrón simple de Rosenblatt (función escalón y regla de aprendizaje del perceptrón) |
| `RedFeedforward` | Perceptrón multicapa con activación sigmoide, salida softmax y retropropagación por mini-lotes |
| `escalar_minmax`, `dividir_datos`, `matriz_confusion`, `exactitud` | Utilidades de preprocesamiento y evaluación |

## Instalación

```bash
pip install git+https://github.com/Crisolis517/redneuronal-cs.git
```

## Uso rápido

```python
import numpy as np
from redneuronal_cs import Perceptron, RedFeedforward

# Compuerta AND con el perceptrón simple
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(Perceptron(eta=0.1, semilla=1).fit(X, [0, 0, 0, 1]).predict(X))   # [0 0 0 1]

# XOR (no lineal) con la red feedforward
red = RedFeedforward(capas_ocultas=(4,), eta=1.0, epocas=3000, lote=4, semilla=0)
print(red.fit(X, [0, 1, 1, 0]).predict(X))                              # [0 1 1 0]
```

## Pruebas

```bash
pip install pytest
pytest -q
```

Licencia MIT.
