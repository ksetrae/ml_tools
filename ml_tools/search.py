from sklearn.model_selection import ParameterGrid
import pandas as pd


class GridSearchSK:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = ParameterGrid(param_grid)
        self.results = None

    def fit(self, x_train, y_train, x_val, y_val):
        grid_len = float(len(self.param_grid))
        progr_report_freq = int(grid_len / 100.) + 1
        curr_results = dict()
        for i, params in enumerate(self.param_grid):
            if i % progr_report_freq == 0:
                print(f'Grid search: {(i / grid_len) * 100}% done')
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            curr_results[i] = {'params': str(params), 'value': self.model.score(x_val, y_val)}

        self.results = pd.DataFrame.from_dict(curr_results, orient='index')

    def print_n_best(self, n):
        print(f'{n} best results:')
        if self.results is None:
            raise AttributeError('Grid search was not fitted, use fit() method')
        if n > self.results.shape[0]:
            n = self.results.shape[0]
        df_cut = self.results.sort_values(by='value', ascending=False)[:n]
        for i, params, value in zip(df_cut.index, df_cut['params'], df_cut['value']):
            print(f'{i}: {params} with value of {value}')


if __name__ == '__main__':
    import random

    from sklearn.ensemble import RandomForestClassifier

    PARAM_GRID = {'n_estimators': [2, 25], 'max_depth': [1, 10], 'criterion': ['gini']}
    df = pd.DataFrame(
        {
            'a': range(1, 101),
            'b': [b * random.random() for b in range(1, 101)],
            'c': [1 if c < 50 else 0 for c in range(1, 101)]
        }
    )
    df = df.sample(frac=1)
    X = df[['a', 'b']]
    y = df['c']

    clf = RandomForestClassifier(random_state=7, n_jobs=-1)
    gs = GridSearchSK(clf, PARAM_GRID)
    gs.fit(X[:70], y[:70], X[70:], y[70:])

    gs.print_n_best(n=2)
