import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入各组成员模型
from models.svm_regression import run_svr
from models.knn_regression import run_knn
# from models.ann_regression import run_ann
# from models.linear_regression import run_linear
# from models.rf_regression import run_rf

def main():
    print('--- Loading Data ---')
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('\n--- SVM Regression ---')
    svr_metrics = run_svr(X_train_scaled, X_test_scaled, y_train, y_test)

    print('\n--- KNN Regression ---')
    knn_metrics = run_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    # print('\n--- ANN Regression ---')
    # ann_metrics = run_ann(X_train_scaled, X_test_scaled, y_train, y_test)
    # print('\n--- Linear Regression ---')
    # linear_metrics = run_linear(X_train_scaled, X_test_scaled, y_train, y_test)
    # print('\n--- Random Forest Regression ---')
    # rf_metrics = run_rf(X_train, X_test, y_train, y_test)  # RF通常不强制归一化

    print('\n--- Final Comparison ---')
    print('| Model | MSE | RMSE | R2 |')
    print('|-------|------|------|----|')
    print(f"| SVM   | {svr_metrics['mse']:.4f} | {svr_metrics['rmse']:.4f} | {svr_metrics['r2']:.4f} |")
    print(f"| KNN   | {knn_metrics['mse']:.4f} | {knn_metrics['rmse']:.4f} | {knn_metrics['r2']:.4f} |")


if __name__ == "__main__":
    main()
