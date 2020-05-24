import random

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn_search_tools.search import GridSearchSK

param_grid = {'n_estimators': [2, 25], 'max_depth': [1, 10], 'criterion': ['gini']}

df = pd.DataFrame(
    {'a': range(1, 101),
     'b': [b * random.random() for b in range(1, 101)],
     'c': [1 if c < 50 else 0 for c in range(1, 101)]})
df = df.sample(frac=1)
X = df[['a', 'b']]
y = df['c']

clf = RandomForestClassifier(random_state=7, n_jobs=-1)

gs = GridSearchSK(clf, param_grid)
gs.fit(X[:70], y[:70], X[70:], y[70:])

gs.print_n_best(n=2)