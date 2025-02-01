import pandas as pd
import numpy as np
from scipy.special import softmax


mu = pd.read_pickle('mu.pickle')
sigma = pd.read_pickle('sigma.pickle')
lamb = pd.read_pickle('lamb.pickle')
lamb = softmax(lamb, axis=1)


m = mu * np.log(10 ** lamb)
V = (sigma * np.log(10 ** lamb)) ** 2
mean = np.exp(m + V/2)
variance = np.exp(2 * m + V) * (np.exp(V) - 1)

Mean = mean.prod(axis=1)
SD = np.sqrt(variance.prod(axis=1) + variance[:, 0] * mean[:, 1] ** 2 + variance[:, 1] * mean[:, 0] ** 2)