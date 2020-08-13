import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import fastipca as ipca


def build_complex(n_chars=11, n_stocks=300, n_dates=750, n_factors=2):
    stocks = pd.Index([f"S{k}" for k in range(n_stocks)])
    chars = pd.Index([f"C{k}" for k in range(n_chars)])
    dates = pd.date_range("2000", "2050")[:n_dates]
    factors = pd.Index([f"F{k}" for k in range(n_factors)])

    P = pd.DataFrame(index=dates, columns=stocks, dtype=bool)
    a = np.random.randint(0, len(dates), P.size // 4)
    b = np.random.randint(0, len(stocks), P.size // 4)

    P.values[a, b] = False
    index = P[P].stack().index

    Z = pd.DataFrame(np.random.randn(len(index), n_chars), index=index, columns=chars)
    R = pd.Series(np.random.randn(len(index)), index=index)
    F = pd.DataFrame(np.random.randn(n_dates, n_factors), index=dates, columns=factors)
    return Z, R, F


def build_simple():
    from statsmodels.datasets import grunfeld

    data = grunfeld.load_pandas().data
    data.year = data.year.astype(int)
    data = data.set_index(["year", "firm"]).sort_index()
    R = data["invest"]
    Z = data.loc[:, "value":]

    F = pd.DataFrame(columns=["exog_1", "exog_2"], index=Z.index.levels[0], dtype=float)
    F[:] = np.random.randn(len(F), 2)
    return Z, R, F


def test1():
    Z, R, F = build_simple()
    gamma, factors = ipca.train(Z, R, 1)
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


def test2():
    Z, R, F = build_simple()
    gamma, factors = ipca.train(Z, R, 1, intercept=True)
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


def test3():
    Z, R, F = build_simple()
    gamma, factors = ipca.train(Z, R, 2, intercept=False)
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


def test4():
    Z, R, F = build_simple()
    gamma, factors = ipca.train(Z, R, 1, exog_factors=F.iloc[:, :1])
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


def test5():
    Z, R, F = build_simple()
    gamma, factors = ipca.train(Z, R, 0, exog_factors=F)
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


def test6():
    Z, R, F = build_complex()
    gamma, factors = ipca.train(Z, R, 1)
    yhat = ipca.predict(Z, gamma, factors)
    yhat_mean = ipca.predict(Z, gamma, factors.mean())
    r2, r2m = r2_score(R, yhat), r2_score(R, yhat_mean)
    print(r2, r2m)


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
