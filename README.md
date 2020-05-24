Tools for tuning the hyper-parameters that are based on scikit-learn modules, 
but have different interface and output style (more convenient, in my opinion).  

### List of implemented modules
- GridSearch

### Install
```
pip install git+https://github.com/ksetrae/sklearn_search_tools.git
```
### Usage example:  
_Set the data and the model type_  
```python
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
```

_Perform the search_
```python
gs = GridSearchSK(clf, param_grid)
gs.fit(X[:70], y[:70], X[70:], y[70:])
```

_Show the results_
```python
gs.print_n_best(n=2)
```

GridSearchSK object also has full results accessible as: 
```python
gs.results
```