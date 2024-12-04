import numpy as np
from scipy import stats
import pandas as pd
import os

# Generate mixture of two normal distributions
x = np.linspace(-5, 5, 1000)
dist = 0.7 * stats.norm.pdf(x, -1, 0.8) + 0.3 * stats.norm.pdf(x, 2, 1.2)
dist = dist / np.sum(dist)

# Save to CSV
path = os.getcwd()
pd.Series(dist).to_csv(path, index=False, header=False)