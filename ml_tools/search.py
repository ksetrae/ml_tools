from sklearn.model_selection import ParameterGrid
import pandas as pd

class GridSearchSK():
    def __init__(self, model, param_grid):
        self.model = model
        self.pg = ParameterGrid(param_grid)

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=True):
        grid_len = float(len(self.pg))
        print_every_iter = int(grid_len / 100.) + 1
        self.results = dict()
        for i, params in enumerate(self.pg):
            if  i % print_every_iter == 0:
                print(f'Grid search: {i/grid_len}% done')
            for param in params:
                setattr(self.model, param, params[param]) 
            self.model.fit(X_train, Y_train)
            self.results[i] = {'params': str(params), 'value': self.model.score(X_val, Y_val)}

        self.results = pd.DataFrame.from_dict(
            self.results, orient='index').sort_values(by='value', ascending=False)

    def print_n_best(self, n):
        if self.results is None:
            raise AttributeError('Grid search was not fitted, use fit() method')
        if n > self.results.shape[0]:
            n = self.results.shape[0]
        for i, row in self.results[:n].iterrows():
            print(f'{i}: {row["params"]} with value of {row["value"]}')


if __name__ == '__main__':
    import random

    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd


    param_grid = {'n_estimators': [2, 25], 'max_depth': [1, 10], 'criterion': ['gini']}
    df = pd.DataFrame(
        {
        # 'a': range(1, 101), 
        'b': [b*random.random() for b in range(1, 101)], 
        'c': [1 if c < 50 else 0 for c in range(1, 101)]
        }
    )
    df = df.sample(frac=1)
    X = df[['b']]
    y = df['c']

    clf = RandomForestClassifier(random_state=7, n_jobs=-1)
    gs = GridSearchSK(clf, param_grid)
    gs.fit(X[:70], y[:70], X[70:], y[70:])

    print(gs.results)
    print()
    gs.print_n_best(2)