import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


def compare_rmse(file_name: str):
    df = pd.read_csv(file_name)

    predictors = ['Year', 'Engine HP', 'Engine Cylinders','highway MPG', 'city mpg', 'Popularity']
    target_col = 'MSRP'

    X = df[predictors].copy()
    y = df[target_col].copy()

    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse_lr = root_mean_squared_error(y_test, y_pred)

    y_dummy = [y_train.mean()] * len(y_test)
    rmse_dummy = root_mean_squared_error(y_test, y_dummy)

    diff = rmse_dummy - rmse_lr

    print(rmse_lr)
    print(rmse_dummy)
    print(diff)

    return rmse_lr, rmse_dummy, diff
