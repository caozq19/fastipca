from sklearn.metrics import r2_score
from statsmodels.datasets import grunfeld
import fastipca as ipca

data = grunfeld.load_pandas().data
data = data.set_index(["year", "firm"])
data = data.sort_index()

X, Y = data.loc[:, "value":], data["invest"]
gamma, F = ipca.train(X, Y)

Y_hat = ipca.predict(X, gamma, F)
print('R2', r2_score(Y, Y_hat))
