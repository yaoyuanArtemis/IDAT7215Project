

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data=pd.read_csv('california_housing.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
columns=data.columns[:-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print('X_train.shape',X_train.shape)
print('X_test.shape',X_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)


model = LinearRegression()

param_grid = {
    'fit_intercept': [True, False],
}

grid=GridSearchCV(model,param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)

best_params=grid.best_params_
print("网格搜索的最佳参数：",best_params)

best_model = grid.best_estimator_

from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)


r2_score_train = best_model.score(X_train, y_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100


r2_score_test = best_model.score(X_test, y_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100


print("训练集评价指标：")
print("R2:", r2_score_train)
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAE:", mae_train)
print("MAPE:", mape_train, "%")

print("\n测试集评价指标：")
print("R2:", r2_score_test)
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("MAPE:", mape_test, "%")







