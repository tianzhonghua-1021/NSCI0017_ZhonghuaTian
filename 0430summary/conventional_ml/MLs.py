import pandas as pd
import numpy as np
import shap
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from scipy.stats import norm

# ============ CONFIGURATION ============
# choose the type: "phys_T,P_TDA", "phys", "T,P", "TDA", "phys_T,P", "phys_TDA"
ANALYSIS_MODE = "phys_TDA"

# mode config settings
MODE_CONFIG = {
    "phys_T,P_TDA": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['G', 'E', 'theta', 'FileID'],
        "output_dir": os.path.join(os.path.dirname(__file__), "phys_T,P_TDA"),
    },
    "phys": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['G', 'E', 'theta', 'FileID', 'TDA_H0_max', 'TDA_H0_min', 'TDA_H0_mean', 'TDA_H0_std', 'TDA_H0_sum', 'TDA_H1_max', 'TDA_H1_min', 'TDA_H1_mean', 'TDA_H1_std', 'TDA_H1_sum', 'TDA_H2_max', 'TDA_H2_min', 'TDA_H2_mean', 'TDA_H2_std', 'TDA_H2_sum', 'Temperature', 'Pressure'],
        "output_dir": os.path.join(os.path.dirname(__file__), "phys"),
    },
    "T,P": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['FileID','E','G','length','n','m','diameter','Ti_count','FG_C=O','FG_NH2','FG_SO3H','FG_none','TiType_1Ti_substitution','TiType_1Ti_surface','TiType_2Ti_substitution','TiType_2Ti_surface', 'TDA_H0_max', 'TDA_H0_min', 'TDA_H0_mean', 'TDA_H0_std', 'TDA_H0_sum', 'TDA_H1_max', 'TDA_H1_min', 'TDA_H1_mean', 'TDA_H1_std', 'TDA_H1_sum', 'TDA_H2_max', 'TDA_H2_min', 'TDA_H2_mean', 'TDA_H2_std', 'TDA_H2_sum','TiType_none','theta'],
        "output_dir": os.path.join(os.path.dirname(__file__), "T,P"),
    },
    "TDA": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['FileID','Temperature','E','G','Pressure','length','n','m','diameter','Ti_count','FG_C=O','FG_NH2','FG_SO3H','FG_none','TiType_1Ti_substitution','TiType_1Ti_surface','TiType_2Ti_substitution','TiType_2Ti_surface', 'TiType_none','theta'],
        "output_dir": os.path.join(os.path.dirname(__file__), "TDA"),
    },
    "phys_T,P": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['G', 'E', 'theta', 'FileID', 'TDA_H0_max', 'TDA_H0_min', 'TDA_H0_mean', 'TDA_H0_std', 'TDA_H0_sum', 'TDA_H1_max', 'TDA_H1_min', 'TDA_H1_mean', 'TDA_H1_std', 'TDA_H1_sum', 'TDA_H2_max', 'TDA_H2_min', 'TDA_H2_mean', 'TDA_H2_std', 'TDA_H2_sum'],
        "output_dir": os.path.join(os.path.dirname(__file__), "phys_T,P"),
    },
    "phys_TDA": {
        "csv_dir": os.path.join(os.path.dirname(__file__), "dataset_full_feat.csv"),
        "drop_cols": ['G','E', 'theta','FileID', 'Temperature', 'Pressure'],
        "output_dir": os.path.join(os.path.dirname(__file__), "phys_TDA"),
    },
}

# ============ MAIN CONFIGURATION ============
config = MODE_CONFIG[ANALYSIS_MODE]
csv_dir = config["csv_dir"]
drop_cols = config["drop_cols"]
output_dir = config["output_dir"]

# output directory setup
os.makedirs(output_dir, exist_ok=True)

# ============ FUNCTIONS ============

def analyze_model_uncertainty(model, X_train, y_train, X_test, y_test, n_iterations=20, model_name="Model"):
    print(f"Performing Bootstrap Uncertainty Analysis for {model_name}...")
    boot_preds = []
    
    for i in range(n_iterations):
        X_resample, y_resample = resample(X_train, y_train, random_state=i)
        
        from sklearn.base import clone
        model_copy = clone(model)
        model_copy.fit(X_resample, y_resample)
        boot_preds.append(model_copy.predict(X_test))
        
    boot_preds = np.array(boot_preds)
    std_dev = np.std(boot_preds, axis=0)
    mean_pred = np.mean(boot_preds, axis=0)
    
    plt.figure(figsize=(5, 5), dpi=300)
    sc = plt.scatter(y_test, mean_pred, c=std_dev, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.colorbar(sc, label='Prediction Uncertainty (Std Dev)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Uncertainty Analysis: {model_name}')
    plt.xlabel('Measured Theta')
    plt.ylabel('Mean Predicted Theta')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_uncertainty_analysis.png'))
    plt.close()
    
    return std_dev


def run_shap_analysis(model, X_test_scaled, feature_names, model_name="Model"):
    X_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    print(f"Calculating {model_name} for SHAP values")
    
    if "RandomForest" in str(type(model)) or "XGBRegressor" in str(type(model)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
    else:
        X_summary = shap.sample(X_df, 50) 
        explainer = shap.KernelExplainer(model.predict, X_summary)
        shap_values = explainer.shap_values(X_summary)
        X_df = X_summary
    
    # Beeswarm Plot
    plt.figure(figsize=(5, 5), dpi=300)
    shap.summary_plot(shap_values, X_df, plot_type="dot", show=False)
    plt.title(f"SHAP Beeswarm Plot: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_shap_beeswarm.png'), bbox_inches='tight')
    plt.close()
    
    # Bar Plot
    plt.figure(figsize=(5, 5), dpi=300)
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_shap_bar.png'), bbox_inches='tight')
    plt.close()
    
    return shap_values


def plot_correlation_matrix(df, model_name="Dataset"):
    corr_matrix = df.corr() 
    plt.figure(figsize=(15, 15), dpi=300)
    sns.heatmap(
        corr_matrix, 
        mask=None, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.2,
        annot_kws={"size": 8}, 
        cbar_kws={'shrink': .4}
    )
    plt.title(f"Pearson Correlation Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_correlation_matrix.png'), bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved: {model_name}_correlation_matrix.png")


def remove_high_correlation_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"There are {len(to_drop)} features (r > {threshold}):")
    print(to_drop)
    X_dropped = X.drop(columns=to_drop)
    return X_dropped, to_drop

def plot_error_rate_vs_significance(uncertainty_results, y_test, predictions_dict):
    """
    Function to plot Error Rate vs Significance for 4 models.
    Significance levels from 0.0 to 1.0 with step 0.2.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    significance_levels = np.arange(0.0, 1.1, 0.2)
    
    for model_name, std_dev in uncertainty_results.items():
        error_rates = []
        y_pred = predictions_dict[model_name]
        abs_error = np.abs(y_test - y_pred)
        
        for alpha in significance_levels:
            if alpha <= 0:
                error_rates.append(0.0)
            elif alpha >= 1.0:
                error_rates.append(1.0)
            else:
                # Calculate Z-score for the given significance level (two-tailed)
                z_score = norm.ppf(1 - alpha / 2)
                # An error occurs if the actual value is outside the prediction interval
                out_of_bounds = abs_error > (z_score * std_dev)
                error_rates.append(np.mean(out_of_bounds))
        
        plt.plot(significance_levels, error_rates, marker='o', label=model_name)

    # Plot the ideal calibration line where Error Rate == Significance
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal (Perfect Calibration)')
    
    plt.title('Uncertainty Analysis: Error Rate vs Significance')
    plt.xlabel('Significance Level')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure according to the logical path[cite: 1]
    save_path = os.path.join(output_dir, 'error_rate_vs_significance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Error rate vs significance plot saved to: {save_path}")

def analyze_model_uncertainty(model, X_train, y_train, X_test, y_test, n_iterations=15, model_name="Model"):
    """
    Perform Bootstrap to estimate prediction uncertainty (std dev)[cite: 1].
    """
    print(f"Performing Bootstrap Uncertainty Analysis for {model_name}...")
    boot_preds = []
    
    for i in range(n_iterations):
        X_resample, y_resample = resample(X_train, y_train, random_state=i)
        from sklearn.base import clone
        model_copy = clone(model)
        model_copy.fit(X_resample, y_resample)
        boot_preds.append(model_copy.predict(X_test))
        
    boot_preds = np.array(boot_preds)
    std_dev = np.std(boot_preds, axis=0)
    return std_dev

def plot_prediction_intervals(uncertainty_results, y_test, predictions_dict):
    """
    Plot Measured vs Predicted values with 95% confidence intervals (error bars) 
    for all 4 models in a combined figure.
    """
    plt.figure(figsize=(15, 12), dpi=300)
    model_names = list(uncertainty_results.keys())
    
    # Define 95% confidence interval multiplier
    z_95 = 1.96 

    for i, name in enumerate(model_names):
        plt.subplot(2, 2, i + 1)
        
        y_pred = predictions_dict[name]
        std_dev = uncertainty_results[name]
        
        # Plotting error bars: y_pred +/- 1.96 * std_dev
        plt.errorbar(y_test, y_pred, yerr=z_95 * std_dev, fmt='o', 
                     ecolor='gray', elinewidth=0.5, capsize=1, 
                     alpha=0.5, mfc='blue', mec='k', markersize=4, label='Predicted w/ 95% CI')
        
        # Perfect prediction line
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
        
        plt.title(f'Prediction Intervals: {name}')
        plt.xlabel('Measured Theta (DFT)')
        plt.ylabel('Predicted Theta')
        plt.legend(fontsize='small')
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'prediction_intervals_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Prediction intervals plot saved to: {save_path}")

def plot_combined_doa_analysis(X_train_scaled, X_test_scaled, y_test, predictions_dict, output_dir):
    """
    Generate a combined Williams Plot for 4 models.
    X-axis: Leverage (h), Y-axis: Standardized Residuals.
    Saves as 'combined_williams_plot.png'.
    """
    print("Generating Combined DOA (Williams Plot) for all models...")
    
    # 1. Calculate Leverage (Shared by all models since features are the same)
    try:
        xtx_inv = np.linalg.pinv(np.dot(X_train_scaled.T, X_train_scaled))
        leverage_test = np.diag(np.dot(np.dot(X_test_scaled, xtx_inv), X_test_scaled.T))
        
        # Calculate threshold h*
        p = X_train_scaled.shape[1]
        n = X_train_scaled.shape[0]
        h_star = 3 * (p + 1) / n
        
        # 2. Setup Figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
        axes = axes.flatten()
        model_names = list(predictions_dict.keys())
        
        for i, name in enumerate(model_names):
            y_pred = predictions_dict[name]
            # Calculate Standardized Residuals
            residuals = y_test - y_pred
            std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            
            # Scatter plot
            ax = axes[i]
            # Points inside domain (h < h* and |std_res| < 3)
            ax.scatter(leverage_test, std_residuals, color='blue', alpha=0.6, edgecolors='k')
            
            # Draw boundaries
            ax.axvline(x=h_star, color='red', linestyle='--', label=f'h* = {h_star:.3f}')
            ax.axhline(y=3, color='orange', linestyle=':', label='Res. Limit (±3)')
            ax.axhline(y=-3, color='orange', linestyle=':')
            
            ax.set_title(f'Williams Plot: {name}')
            ax.set_xlabel('Leverage (h)')
            ax.set_ylabel('Standardized Residuals')
            ax.legend(fontsize='small')
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'combined_williams_plot.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Combined Williams Plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error in DOA analysis: {e}")

def plot_model_performance_comparison(metrics_df, output_dir):
    """
    绘制4种模型在Train和Test集上的R2和RMSE对比柱状图
    """
    print("Generating Model Performance Comparison Bar Plots...")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 定义指标和标题
    metrics = ['R2', 'RMSE']
    titles = ['R² Score Comparison (Higher is better)', 'RMSE Comparison (Lower is better)']
    
    for i, metric in enumerate(metrics):
        # 准备绘图数据
        plot_data = metrics_df[metrics_df['Metric'] == metric]
        
        # 绘图
        sns.barplot(
            data=plot_data, 
            x='Model', 
            y='Value', 
            hue='Dataset', 
            ax=axes[i],
            palette='muted'
        )
        
        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('Model')
        axes[i].legend(title='Dataset')
        
        # 在柱状图上添加数值标签
        for p in axes[i].patches:
            height = p.get_height()
            # 只有高度大于 1e-6 (避开零值和极小值) 且高度不是 NaN 时才进行标注
            if height > 1e-6 and not np.isnan(height):
                axes[i].annotate(f'{height:.3f}', 
                                (p.get_x() + p.get_width() / 2., height), 
                                ha='center', va='center', 
                                xytext=(0, 9), 
                                textcoords='offset points',
                                fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Performance comparison bar plot saved to: {save_path}")

# ============ MAIN EXECUTION ============
print(f"\n{'='*60}")
print(f"Running analysis mode: {ANALYSIS_MODE}")
print(f"Output directory: {output_dir}")
print(f"{'='*60}\n")

# Load data
df = pd.read_csv(csv_dir)
df.columns = df.columns.str.strip()

# Prepare data
X = df.drop(columns=drop_cols)
y = df['theta']

# Pearson analysis
analysis_df = X.copy()
analysis_df['theta'] = y 
plot_correlation_matrix(analysis_df, model_name="Feature_Analysis")

# Drop highly correlated features
X, dropped_cols = remove_high_correlation_features(X, threshold=0.9)
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Define and train models
rf = RandomForestRegressor(random_state=42)
svr = SVR()
lr = LinearRegression()
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
svr_param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2], 'kernel': ['rbf']}
xgb_param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6], 'subsample': [0.8, 1.0]}

grid_rf = GridSearchCV(rf, rf_param_grid, cv=5)
grid_rf.fit(X_train_scaled, y_train)

grid_svr = GridSearchCV(svr, svr_param_grid, cv=5)
grid_svr.fit(X_train_scaled, y_train)

grid_xgb = GridSearchCV(xgb, xgb_param_grid, cv=5)
grid_xgb.fit(X_train_scaled, y_train)

lr.fit(X_train_scaled, y_train)

best_rf = grid_rf.best_estimator_
best_svr = grid_svr.best_estimator_
best_xgb = grid_xgb.best_estimator_

# Make predictions
y_pred_rf = best_rf.predict(X_test_scaled)
y_pred_svr = best_svr.predict(X_test_scaled)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Calculate metrics
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# Visualization
plt.figure(figsize=(15, 15), dpi=300)

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='k', color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Random Forest (R2={r2_rf:.3f} RMSE={rmse_rf:.4f})')
plt.xlabel('Measured Theta')
plt.ylabel('Predicted Theta')

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_svr, alpha=0.6, edgecolors='k', color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'SVR (R2={r2_svr:.3f} RMSE={rmse_svr:.4f})')
plt.xlabel('Measured Theta')
plt.ylabel('Predicted Theta')

plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_lr, alpha=0.6, edgecolors='k', color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'Linear Regression (R2={r2_lr:.3f} RMSE={rmse_lr:.4f})')
plt.xlabel('Measured Theta')
plt.ylabel('Predicted Theta')

plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_xgb, alpha=0.6, edgecolors='k', color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title(f'XGBoost (R2={r2_xgb:.3f} RMSE={rmse_xgb:.4f})')
plt.xlabel('Measured Theta')
plt.ylabel('Predicted Theta')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
plt.close()
print(f"✓ Model comparison saved")

# Uncertainty analysis
models_to_analyze = [
    (best_rf, "Random_Forest"),
    (best_svr, "SVR"),
    (lr, "Linear_Regression"),
    (best_xgb, "XGBoost")
]

uncertainty_results = {}
predictions_dict = {}

for model_obj, name in models_to_analyze:
    std_uncertainty = analyze_model_uncertainty(
        model=model_obj, 
        X_train=X_train_scaled, 
        y_train=y_train, 
        X_test=X_test_scaled, 
        y_test=y_test,
        n_iterations=15,
        model_name=name
    )
    uncertainty_results[name] = std_uncertainty
    predictions_dict[name] = model_obj.predict(X_test_scaled)

# SHAP analysis
shap_values_rf = run_shap_analysis(best_rf, X_test_scaled, X.columns, model_name="Random_Forest")
shap_values_svr = run_shap_analysis(best_svr, X_test_scaled, X.columns, model_name="SVR")
shap_values_lr = run_shap_analysis(lr, X_test_scaled, X.columns, model_name="Linear_Regression")
shap_values_xgb = run_shap_analysis(best_xgb, X_test_scaled, X.columns, model_name="XGBoost")

# Error rate vs significance plot
plot_error_rate_vs_significance(uncertainty_results, y_test.values, predictions_dict)
# prediction intervals plot
plot_prediction_intervals(uncertainty_results, y_test.values, predictions_dict)
# Perform combined DOA (Williams Plot) analysis
plot_combined_doa_analysis(
    X_train_scaled=X_train_scaled, 
    X_test_scaled=X_test_scaled, 
    y_test=y_test.values, 
    predictions_dict=predictions_dict, 
    output_dir=output_dir
)
# Performance comparison bar plot
# 1. 计算所有模型在训练集上的表现（用于对比）
y_train_pred_rf = best_rf.predict(X_train_scaled)
y_train_pred_svr = best_svr.predict(X_train_scaled)
y_train_pred_lr = lr.predict(X_train_scaled)
y_train_pred_xgb = best_xgb.predict(X_train_scaled)

# 2. 整理数据用于绘图
performance_data = []
models_list = [
    ("Random Forest", y_train_pred_rf, y_pred_rf),
    ("SVR", y_train_pred_svr, y_pred_svr),
    ("Linear Regression", y_train_pred_lr, y_pred_lr),
    ("XGBoost", y_train_pred_xgb, y_pred_xgb)
]

for name, train_pred, test_pred in models_list:
    # 训练集指标
    performance_data.append([name, 'Train', 'R2', r2_score(y_train, train_pred)])
    performance_data.append([name, 'Train', 'RMSE', np.sqrt(mean_squared_error(y_train, train_pred))])
    # 测试集指标
    performance_data.append([name, 'Test', 'R2', r2_score(y_test, test_pred)])
    performance_data.append([name, 'Test', 'RMSE', np.sqrt(mean_squared_error(y_test, test_pred))])

metrics_df = pd.DataFrame(performance_data, columns=['Model', 'Dataset', 'Metric', 'Value'])

# 3. 调用绘图函数
plot_model_performance_comparison(metrics_df, output_dir)

print(f"\n{'='*60}")
print(f"Analysis complete! All results saved to:")
print(f"{output_dir}")
print(f"{'='*60}\n")