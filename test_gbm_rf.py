import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pdb


def make_sample():
    """
    Return (X_train, X_test, y_train, y_test)
    """
    X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    result = (
        X_train,
        X_test,
        y_train,
        y_test
    )

    return result


def csv_to_array(handle):
    """
    Return (X_train, X_test, y_train, y_test)
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    with open(handle, 'r') as fi:
        # Remove title
        fi.readline()

        for line in fi:
            data = line.rstrip('\n').split(',')
            assert data[0] in ('1', '0'), 'The First column must be 1 or 0.'

            for i, item in enumerate(data):
                if item == 'NaN':
                    # data[i] = np.nan
                    data[i] = 0
                else:
                    try:
                        data[i] = float(item)
                    except ValueError:
                        Exception('Format error.')

            if data[0] == 1:
                # is training data
                X_train.append(data[13:])
                y_train.append(data[1:13])
            else:
                # is testing data
                X_test.append(data[13:])
                y_test.append(data[1:13])

    result = (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
    )

    return result


if __name__ == '__main__':
    import sys

    # Load data
    is_userdefined = False
    if len(sys.argv) == 1:
        print('Load sample data...')
        X_train, X_test, y_train, y_test = make_sample()
    else:
        is_userdefined = True
        print('Load file: \'{}\''.format(sys.argv[1]))
        X_train, X_test, y_train, y_test = csv_to_array(sys.argv[1])

    # GBM regressor
    gbm_params = {
        'n_estimators': 500,
        'max_depth': 4,
        'min_samples_split': 1,
        'learning_rate': 0.01,
        'loss': 'ls'
    }

    gbm_clf = GradientBoostingRegressor(**gbm_params)

    if is_userdefined:
        gbm_clf.fit(X_train, zip(*y_train)[0])  # Outcom of first month in y_train
        gbm_mse = mean_squared_error(zip(*y_test)[0], gbm_clf.predict(X_test))  # Outcom of first month in y_test
    else:
        gbm_clf.fit(X_train, y_train)
        gbm_mse = mean_squared_error(y_test, gbm_clf.predict(X_test))

    print('GBM Regressor MSE: {:.4f}'.format(gbm_mse))

    # RF regressor
    rf_params = {
        'n_estimators': 500,
        'max_depth': 4,
        'min_samples_split': 1,
    }

    rf_clf = RandomForestRegressor(**rf_params)
    rf_clf.fit(X_train, y_train)
    rf_mse = mean_squared_error(y_test, rf_clf.predict(X_test))

    print('RF Regressor MSE: {:.4f}'.format(rf_mse))
