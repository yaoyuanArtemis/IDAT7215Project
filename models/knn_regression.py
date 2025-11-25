import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def _plot_pred_vs_true(y_test, y_pred, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "knn_pred_vs_true.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)

    # Ideal y = x reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True Median House Value")
    plt.ylabel("Predicted Value")
    plt.title("KNN Regression: Prediction vs True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved KNN prediction plot to {save_path}")


def _plot_residuals(y_test, y_pred, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "knn_residuals.png")

    residuals = y_test - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Count")
    plt.title("KNN Regression Residual Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved KNN residual plot to {save_path}")

def run_knn(X_train_scaled, X_test_scaled, y_train, y_test, do_plots=False):


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


    if do_plots:
        _plot_pred_vs_true(y_test, y_pred)
        _plot_residuals(y_test, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "best_params": gs.best_params_
    }
run_knn