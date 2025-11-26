import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_linear(X_train, X_test, y_train, y_test):
    """
    执行线性回归模型 (带网格搜索)
    """
    
    # 1. 定义模型和参数网格
    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
    }

    # 2. 网格搜索训练
    grid = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    # 3. 获取最佳模型和参数
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    print(f"Linear Best Params: {best_params}")

    # 4. 预测
    y_pred_test = best_model.predict(X_test)

    # 5. 计算指标 (保留你原有的丰富指标打印，方便调试)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_score_test = r2_score(y_test, y_pred_test)
    
    # 防止除以0警告，加一个微小值，或者简单处理
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print(f"Linear Test Metrics -> R2: {r2_score_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

    # 6. 返回 main.py 需要的字典格式
    return {
        'mse': mse_test,
        'rmse': rmse_test,
        'r2': r2_score_test
    }
