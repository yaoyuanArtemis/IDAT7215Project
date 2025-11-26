# Group 13: Machine Learning Regression Model Comparison

## Project Overview

This project presents a systematic comparison of five machine learning regression algorithms on the California Housing dataset ([fetch_california_housing from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)). Our goal is to analyze each model’s prediction performance and discuss their applicability to real-world housing price estimation.

## Author
| Name        | Student ID | Model Assigned                  |
| ----------- | ---------- | ------------------------------- |
| Zhou Wei  | 3036605167   | Linear Regression  |
| Liu Feng   | 3036509117   | Support Vector Machine (SVM)    |
| Liang Xudong | 3036605832   | K-Nearest Neighbors (KNN)       |
| Wangjunjie  | 3036575295   | Random Forest   |
Zhang Gaoxiang   | 3036507779   | Artificial Neural Network (ANN) |

## How to Run

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```
    python main.py
    ```
3. Outputs, including metrics and charts, will be found in the `results/` directory.

## Data Source

- **Dataset:** California Housing (scikit-learn builtin)
- **Features:** 8 numerical features per record
- **Target:** Median house value (continuous variable)

## Models Evaluated

Each team member selected and implemented one regression algorithm:


- **Linear Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Artificial Neural Network (ANN)**


## Workflow

1. **Data Preprocessing:** Clean and normalize the raw California Housing dataset.
2. **Model Implementation:** Train each algorithm independently using standard Python libraries (scikit-learn, TensorFlow/Keras as needed).
3. **Evaluation:** Use metrics such as MAE, RMSE, and R² to compare prediction performance.
4. **Result Visualization:** Summarize findings with charts and tables.