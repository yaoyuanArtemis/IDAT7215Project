import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 导入各组成员模型
from models.svm_regression import run_svr
from models.knn_regression import run_knn
from models.ann_regression import run_ann
from models.linear_regression import run_linear
from models.rf_regression import run_rf

def main():

    # 1. Data Preprocessing
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

    # 2. Model Implementation

    print('\n--- Linear Regression ---')
    linear_metrics = run_linear(X_train_scaled, X_test_scaled, y_train, y_test)

    print('\n--- SVM Regression ---')
    svr_metrics = run_svr(X_train_scaled, X_test_scaled, y_train, y_test)

    print('\n--- KNN Regression ---')
    knn_metrics = run_knn(X_train_scaled, X_test_scaled, y_train, y_test)

    print('\n--- Random Forest Regression ---')
    rf_metrics = run_rf(X_train,X_test, y_train, y_test)

    print('\n--- ANN Regression ---')
    ann_metrics = run_ann(X_train_scaled, X_test_scaled, y_train, y_test)
    
    

    # 3. Evaluation
    print('\n--- Final Comparison ---')
    print('| Model | MSE | RMSE | R2 |')
    print('|-------|------|------|----|')
    print(f"| LR   | {linear_metrics['mse']:.4f} | {linear_metrics['rmse']:.4f} | {linear_metrics['r2']:.4f} |")
    print(f"| SVM   | {svr_metrics['mse']:.4f} | {svr_metrics['rmse']:.4f} | {svr_metrics['r2']:.4f} |")
    print(f"| KNN   | {knn_metrics['mse']:.4f} | {knn_metrics['rmse']:.4f} | {knn_metrics['r2']:.4f} |")
    print(f"|  RF   | {rf_metrics['mse']:.4f} | {rf_metrics['rmse']:.4f} | {rf_metrics['r2']:.4f} |")
    print(f"| ANN   | {ann_metrics['mse']:.4f} | {ann_metrics['rmse']:.4f} | {ann_metrics['r2']:.4f} |")

    # 4. Visualization
    models = ['LR', 'SVM', 'KNN', 'RF', 'ANN']
    rmse_scores = [
        linear_metrics['rmse'], 
        svr_metrics['rmse'], 
        knn_metrics['rmse'], 
        rf_metrics['rmse'], 
        ann_metrics['rmse']
    ]
    r2_scores = [
        linear_metrics['r2'], 
        svr_metrics['r2'], 
        knn_metrics['r2'], 
        rf_metrics['r2'], 
        ann_metrics['r2']
    ]

    x = np.arange(len(models))  
    width = 0.35 

    fig, ax1 = plt.subplots(figsize=(10, 6))

    rects1 = ax1.bar(x - width/2, rmse_scores, width, label='RMSE (Lower is Better)', color='#FF9999', alpha=0.9)
    ax1.set_ylabel('RMSE Score', fontweight='bold')
    ax1.set_title('Model Comparison: RMSE vs R2 Score', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylim(0, max(rmse_scores) * 1.2)  

    ax2 = ax1.twinx()
    ax2.plot(models, r2_scores, color='#3366CC', marker='o', linewidth=2, markersize=8, label='R2 (Higher is Better)')
    ax2.set_ylabel('R2 Score', fontweight='bold', color='#3366CC')
    ax2.tick_params(axis='y', labelcolor='#3366CC')
    ax2.set_ylim(0, 1.05)  
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    print("\nGenerated comparison chart: 'model_comparison.png'")
    plt.savefig('/results/model_comparison.png', dpi=300) 
    plt.show() 

     # --- 5. Generate Table Image (New Addition) ---
    data = [
        ['Linear Regression', linear_metrics['mse'], linear_metrics['rmse'], linear_metrics['r2']],
        ['SVM Regression', svr_metrics['mse'], svr_metrics['rmse'], svr_metrics['r2']],
        ['KNN Regression', knn_metrics['mse'], knn_metrics['rmse'], knn_metrics['r2']],
        ['Random Forest', rf_metrics['mse'], rf_metrics['rmse'], rf_metrics['r2']],
        ['ANN Regression', ann_metrics['mse'], ann_metrics['rmse'], ann_metrics['r2']]
    ]
    columns = ['Model', 'MSE', 'RMSE', 'R2 Score']
    df = pd.DataFrame(data, columns=columns)

    fig_table, ax_table = plt.subplots(figsize=(10, 4)) # 调整图片大小
    ax_table.axis('tight')
    ax_table.axis('off')
    ax_table.set_title('Model Performance Metrics', fontweight='bold', fontsize=14, y=1.1)

    table = ax_table.table(cellText=df.values,
                           colLabels=df.columns,
                           loc='center',
                           cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2) # 调整行高和列宽 (宽, 高)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#D3D3D3') # 浅灰色边框
        if row == 0:
            cell.set_facecolor('#40466e') # 表头深蓝色背景
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5') # 偶数行浅灰背景
        else:
            cell.set_facecolor('white')

    print("\nGenerated table image: 'model_performance_table.png'")
    plt.savefig('/results/model_performance_table.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
