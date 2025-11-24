import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def run_knn(X_train_scaled, X_test_scaled, y_train, y_test):


    print("Tuning KNN hyperparameters...")

    knn = KNeighborsRegressor()

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1=Manhattan, 2=Euclidean
        "metric": ["minkowski"]
    }

    gs = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train_scaled, y_train)

    best_knn = gs.best_estimator_
    print("Best Params:", gs.best_params_)
    print("Best CV RMSE:", -gs.best_score_)

    # Evaluate on test set
    y_pred = best_knn.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nKNN Test Performance")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "best_params": gs.best_params_
    }
