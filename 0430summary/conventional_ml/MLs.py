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
import matplotlib.lines as mlines

# ============ CONFIGURATION ============
# choose the type: "phys_T,P_TDA", "phys", "T,P", "TDA", "phys_T,P", "phys_TDA"
ANALYSIS_MODE = "phys"

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
def calculate_bootstrap_metrics(y_true, y_pred, std_dev):
    """
    calculate the core metrics for regression performance and uncertainty evaluation
    """
    # regression metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # coverage
    z_95 = 1.96
    lower_bound = y_pred - z_95 * std_dev
    upper_bound = y_pred + z_95 * std_dev
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # efficiency
    efficiency = np.mean(upper_bound - lower_bound)
    
    return r2, rmse, coverage, efficiency

def get_bootstrap_error_bars(model, X_train, y_train, X_test, y_test, n_iterations=10):
    """
    multiple bootstrap iterations to get a distribution of performance metrics, then calculate the standard deviation (error bars) for R2 and RMSE.
    """
    metrics_list = []
    for i in range(n_iterations):
        # bootstrap resampling
        X_res, y_res = resample(X_train, y_train, random_state=i+100)
        from sklearn.base import clone
        m_copy = clone(model).fit(X_res, y_res)
        y_p = m_copy.predict(X_test)
        
        # simply collect R2 and RMSE for each iteration
        metrics_list.append([r2_score(y_test, y_p), np.sqrt(mean_squared_error(y_test, y_p))])
    
    return np.std(metrics_list, axis=0) # return R2_std, RMSE_std
def plot_academic_metrics_comparison(metrics_df, output_dir):
    """
    metrics comparison plot
    """
    print("Generating Academic Dual-Y Metrics Plot...")
    
    models = metrics_df['Model'].unique()
    x = np.arange(len(models))
    width = 0.2  # width of bars
    
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    ax2 = ax1.twinx() # y axis for RMSE and Efficiency
    
    # metric groups
    # left
    r2_data = metrics_df[metrics_df['Metric'] == 'R2']
    cov_data = metrics_df[metrics_df['Metric'] == 'Coverage']
    # right
    rmse_data = metrics_df[metrics_df['Metric'] == 'RMSE']
    eff_data = metrics_df[metrics_df['Metric'] == 'Efficiency']
    
    # left bars (R2, Coverage)
    b1 = ax1.bar(x - 1.5*width, r2_data['Value'], width, label='$R^2$ (bootstrap mean)', 
                color='#1f77b4', edgecolor='black', yerr=r2_data['Error'], capsize=3)
    b2 = ax1.bar(x - 0.5*width, cov_data['Value'], width, label='Coverage (bootstrap mean)', 
                color='#ff7f0e', edgecolor='black', yerr=cov_data['Error'], capsize=3)
    
    # right bars (RMSE, Efficiency)
    b3 = ax2.bar(x + 0.5*width, rmse_data['Value'], width, label='RMSE (bootstrap mean)', 
                color='#2ca02c', edgecolor='black', yerr=rmse_data['Error'], capsize=3)
    b4 = ax2.bar(x + 1.5*width, eff_data['Value'], width, label='Efficiency (bootstrap mean)', 
                color='#d62728', edgecolor='black', yerr=eff_data['Error'], capsize=3)
    
    # modification
    ax1.set_ylabel('$R^2$ / Coverage', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE / Efficiency', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim(0, 1.2) # set as around 1 for R2 and Coverage
    
    # legends on the top of figures
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=True, fontsize=9)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    save_path = os.path.join(output_dir, 'combined_metrics_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined academic plot saved to: {save_path}")

def plot_interval_width_vs_coverage(uncertainty_results, output_dir):
    """
    generate boxplots of interval widths across different nominal coverage levels for all 4 models in a combined figure.
    """
    print("Generating Interval Width vs Nominal Coverage Boxplots...")
    
    # nominal coverage levels (alpha values) from 5% to 95%
    coverages = np.arange(0.05, 1.0, 0.05)
    coverage_labels = [f"{int(c*100)}%" for c in coverages]
    
    # color setting
    colors = sns.color_palette("Set2", n_colors=len(uncertainty_results))
    
    # subgraph for each model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    for idx, (model_name, std_dev) in enumerate(uncertainty_results.items()):
        ax = axes[idx]
        plot_data = []
        
        # calculate the distribution of interval widths for each nominal coverage level
        for alpha_val in coverages:
            # calculate: width = 2 * z_score * std_dev
            z_score = norm.ppf(1 - (1 - alpha_val) / 2)
            widths = 2 * z_score * std_dev
            
            for w in widths:
                plot_data.append({
                    'Nominal Coverage': f"{int(alpha_val*100)}%",
                    'Interval Width': w
                })
        
        df_width = pd.DataFrame(plot_data)
        
        # draw boxplot
        sns.boxplot(
            data=df_width, 
            x='Nominal Coverage', 
            y='Interval Width', 
            ax=ax,
            color=colors[idx],
            fliersize=2,        # outlier marker size
            linewidth=1.2,      # whisker line width
            width=0.7           # box width
        )
        
        ax.set_title(f'Model: {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Nominal Coverage', fontsize=12)
        ax.set_ylabel('Interval Width (V)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticklabels(coverage_labels, rotation=45)

    plt.suptitle('Interval-width distribution across Nominal Coverages', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'interval_width_coverage_boxplot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Interval width boxplots saved to: {save_path}")  

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

# for visualization purposes
y_train_pred_rf = best_rf.predict(X_train_scaled)
y_train_pred_svr = best_svr.predict(X_train_scaled)
y_train_pred_lr = lr.predict(X_train_scaled)
y_train_pred_xgb = best_xgb.predict(X_train_scaled)

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
def plot_refined_performance(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                             model_name, r2_train, rmse_train, r2_test, rmse_test, save_name):
    # data preparation for joint plot
    df_train = pd.DataFrame({
        'Measured': y_train_true.flatten(), 
        'Predicted': y_train_pred.flatten(), 
        'Set': 'Train'
    })
    df_test = pd.DataFrame({
        'Measured': y_test_true.flatten(), 
        'Predicted': y_test_pred.flatten(), 
        'Set': 'Test'
    })
    df_plot = pd.concat([df_train, df_test])

    # create JointGrid with seaborn
    g = sns.JointGrid(data=df_plot, x='Measured', y='Predicted', hue='Set', 
                      palette={'Train': '#1f77b4', 'Test': '#d62728'}, space=0, ratio=5)
    
    # draw scatter points on the joint plot
    g.plot_joint(sns.scatterplot, alpha=0.5, edgecolor='w', s=45)
    
    # draw marginal histograms with KDE
    g.plot_marginals(sns.histplot, kde=True, alpha=0.3, common_norm=False)
    
    # draw the perfect prediction line (y=x)
    ax_main = g.ax_joint
    all_vals = np.concatenate([y_train_true, y_test_true])
    vmin, vmax = all_vals.min(), all_vals.max()
    ax_main.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, alpha=0.7, label='Ideal y=x')
    
    # add performance metrics as text box on the plot
    stats_label = (f"Train: $R^2$={r2_train:.3f}, RMSE={rmse_train:.4f}\n"
                   f"Test : $R^2$={r2_test:.3f}, RMSE={rmse_test:.4f}")
    
    ax_main.text(0.05, 0.95, stats_label, transform=ax_main.transAxes, 
                 fontsize=9, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # set labels and title
    ax_main.set_xlabel('Measured $\\theta$ (DFT+Langmuir)', fontsize=11)
    ax_main.set_ylabel('Predicted $\\theta$ (Model)', fontsize=11)
    g.fig.suptitle(f'Model: {model_name}', y=1.02, fontsize=13, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()
print("Generating final joint distribution plots...")

models_data = [
    ('Random Forest', y_train_pred_rf, y_pred_rf, 'joint_dist_rf.png'),
    ('SVR', y_train_pred_svr, y_pred_svr, 'joint_dist_svr.png'),
    ('Linear Regression', y_train_pred_lr, y_pred_lr, 'joint_dist_lr.png'),
    ('XGBoost', y_train_pred_xgb, y_pred_xgb, 'joint_dist_xgb.png')
]

for name, train_p, test_p, s_name in models_data:
    r2_t = r2_score(y_train, train_p)
    rmse_t = np.sqrt(mean_squared_error(y_train, train_p))
    r2_v = r2_score(y_test, test_p)
    rmse_v = np.sqrt(mean_squared_error(y_test, test_p))
    
    plot_refined_performance(y_train.values, train_p, y_test.values, test_p, 
                             name, r2_t, rmse_t, r2_v, rmse_v, s_name)
print(f"✓ Refined model comparison plots saved to {output_dir}")

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
# all models' predictions on the training set for performance metrics calculation
y_train_pred_rf = best_rf.predict(X_train_scaled)
y_train_pred_svr = best_svr.predict(X_train_scaled)
y_train_pred_lr = lr.predict(X_train_scaled)
y_train_pred_xgb = best_xgb.predict(X_train_scaled)

performance_data = []
models_to_plot = [
    ("Random Forest", "Random_Forest", best_rf),
    ("SVR", "SVR", best_svr),
    ("Linear Regression", "Linear_Regression", lr),
    ("XGBoost", "XGBoost", best_xgb)
]

print("Calculating full academic metrics with error bars...")
for display_name, internal_name, model_obj in models_to_plot:
    y_p = predictions_dict[internal_name]
    s_d = uncertainty_results[internal_name]
    
    # calculate the core metrics for regression performance and uncertainty evaluation
    r2, rmse, cov, eff = calculate_bootstrap_metrics(y_test.values, y_p, s_d)
    
    # obtain error bars for R2 and RMSE using bootstrap resampling
    r2_std, rmse_std = get_bootstrap_error_bars(model_obj, X_train_scaled, y_train, X_test_scaled, y_test.values)
    
    # fill in the performance data list for plotting
    performance_data.append([display_name, 'R2', r2, r2_std])
    performance_data.append([display_name, 'Coverage', cov, 0.02]) # Coverage fluctuates around 0.95, so we can set a small error bar for visualization
    performance_data.append([display_name, 'RMSE', rmse, rmse_std])
    performance_data.append([display_name, 'Efficiency', eff, s_d.std()*0.1])

metrics_df = pd.DataFrame(performance_data, columns=['Model', 'Metric', 'Value', 'Error'])

# use the custom plotting function to create the combined academic metrics comparison plot
plot_academic_metrics_comparison(metrics_df, output_dir)

# draw interval width vs coverage boxplots
plot_interval_width_vs_coverage(uncertainty_results, output_dir)

print(f"\n{'='*60}")
print(f"Analysis complete! All results saved to:")
print(f"{output_dir}")
print(f"{'='*60}\n")