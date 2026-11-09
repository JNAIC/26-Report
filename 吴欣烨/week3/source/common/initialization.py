import numpy as np

def std_initialization(shape: tuple, std: float=0.01) -> np.ndarray:
    return np.random.randn(*shape) * std

def Xavier(shape: tuple) -> np.ndarray:
    return np.random.randn(*shape) * np.sqrt(1.0/shape[0])

def He(shape: tuple) -> np.ndarray:
    return np.random.randn(*shape) * np.sqrt(2.0/shape[0])
