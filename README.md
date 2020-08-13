[![Build Status](https://travis-ci.org/0x0L/fastipca.svg?branch=master)](https://travis-ci.org/0x0L/fastipca)
[![Documentation Status](https://readthedocs.org/projects/fastipca/badge/?version=latest)](https://fastipca.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/0x0L/fastipca/badge.svg?branch=master)](https://coveralls.io/github/0x0L/fastipca?branch=master)

# Instrumented Principal Components Analysis

This is a Python implementation of the Instrumented Principal Components Analysis algorithm by Kelly, Pruitt, Su (2017).

## Usage

```python
from statsmodels.datasets import grunfeld
from sklearn.metrics import r2_score
import fastipca as ipca

data = grunfeld.load_pandas().data
data.year = data.year.astype(int)
data = data.set_index(["year", "firm"])

# IMPORTANT: time must be the first index
# and the panel must be sorted!
data = data.sort_index()

Z = data.loc[:, "value":]
R = data["invest"]

gamma, factors = ipca.train(Z, R)

yhat = ipca.predict(Z, gamma, factors)
print('R2 total', r2_score(R, yhat))

yhat = ipca.predict(Z, gamma, factors.mean())
print('R2 mean factor', r2_score(R, yhat))
```
