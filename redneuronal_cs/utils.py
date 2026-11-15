import numpy as np


def escalar_minmax(X_entrenamiento, *otros):
    """Normaliza a [0, 1] usando el mínimo y máximo del conjunto de entrenamiento."""
    X = np.asarray(X_entrenamiento, float)
    mn, mx = X.min(axis=0), X.max(axis=0)
    rango = np.where(mx - mn == 0, 1, mx - mn)
    f = lambda Z: (np.asarray(Z, float) - mn) / rango
    return (f(X),) + tuple(f(o) for o in otros)


def dividir_datos(X, y, prop_prueba=0.3, semilla=None):
    """Partición aleatoria estratificada en entrenamiento y prueba."""
    X, y = np.asarray(X), np.asarray(y).ravel()
    rng = np.random.default_rng(semilla)
    te = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        te.extend(rng.choice(i, int(round(prop_prueba * len(i))), replace=False))
    te = np.array(sorted(te)); tr = np.setdiff1d(np.arange(len(y)), te)
    return X[tr], X[te], y[tr], y[te]


def matriz_confusion(y_real, y_pred):
    """Matriz de confusión (filas = real, columnas = predicho)."""
    clases = np.unique(np.concatenate([y_real, y_pred]))
    m = np.zeros((len(clases), len(clases)), int)
    for r, p in zip(y_real, y_pred):
        m[np.searchsorted(clases, r), np.searchsorted(clases, p)] += 1
    return m


def exactitud(y_real, y_pred):
    return float(np.mean(np.asarray(y_real) == np.asarray(y_pred)))
