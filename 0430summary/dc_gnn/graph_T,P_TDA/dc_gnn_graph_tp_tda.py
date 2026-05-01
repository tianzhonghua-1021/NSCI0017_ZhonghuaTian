import os
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
# import shutil
import random
from deepchem.feat.graph_data import GraphData
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import shap
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DGLBACKEND'] = 'pytorch'
# 1. path setting
csv_folder = '/home/ucaqzti/deepchem/data_prepare'
main_csv = '/home/ucaqzti/models/dataset_form.csv'
dataset_cache_dir = './dmpnn_dataset_cache'  # cache folder


def compute_tda_ripser_15d(mol):
    try:
        coords = mol.GetConformer().GetPositions()
        result = ripser(coords, maxdim=1)
        dgms = result['dgms']
        h0, h1 = dgms[0], dgms[1]
        h0_lifetimes = sorted(h0[:, 1] - h0[:, 0], reverse=True)
        h0_feat = h0_lifetimes[1:8] if len(h0_lifetimes) > 1 else []
        h1_lifetimes = sorted(h1[:, 1] - h1[:, 0], reverse=True)
        h1_feat = h1_lifetimes[:7]
        if len(h0_feat) < 7: h0_feat += [0.0] * (7 - len(h0_feat))
        if len(h1_feat) < 7: h1_feat += [0.0] * (7 - len(h1_feat))
        h1_count = [float(len(h1))]
        
        return np.concatenate([h0_feat, h1_feat, h1_count]).astype(np.float32)
    except Exception as e:
        return np.zeros(15, dtype=np.float32)

def csv_to_rdkit_mol_safe(file_path):
    try:
        df_mol = pd.read_csv(file_path).dropna()
        atom_count = len(df_mol)
        xyz_lines = [str(atom_count), "compatibility_fix"]
        for _, row in df_mol.iterrows():
            symbol = str(row['Element']).strip()
            x, y, z = [f"{float(val):.10f}" for val in [row['X'], row['Y'], row['Z']]]
            xyz_lines.append(f"{symbol} {x} {y} {z}")
        
        xyz_block = "\n".join(xyz_lines) + "\n"
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None: return None
        
        new_mol = Chem.Mol(mol)
        # try the connection
        try:
            from rdkit.Chem import rdDetermineBonds
            rdDetermineBonds.DetermineConnectivity(new_mol)
        except:
            pass 
        return new_mol
    except:
        return None
# learning curve plot
def plot_learning_curve(losses):
    plt.figure(figsize=(5, 5),dpi=300)
    plt.plot(losses, label='Training Loss', color='#1f77b4', alpha=0.3, linestyle='-')
    if len(losses) > 5:
        smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
        plt.plot(range(4, len(losses)), smooth_losses, label='Smoothed Loss (Moving Avg)', color='#d62728', lw=2)
    plt.title('DMPNN Training Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "learning_curve.png")
    plt.savefig(save_path)

# SHAP analysis
def shap_analysis(model, dataset, transformers):
    print("Start SHAP analysis")
    
    # 1. Global features for SHAP
    all_global_feats = []
    for i in range(len(dataset)):
        all_global_feats.append(dataset.X[i].global_features)
    
    X_global = np.array(all_global_feats)
    phys_names = ['Temperature (K)', 'G', 'Pressure (bar)', '1/T', 'log(P)']
    tda_names = [f'TDA_H0_{i}' for i in range(7)] + \
                [f'TDA_H1_{i}' for i in range(7)] + \
                ['TDA_H1_Count']
    feature_names = phys_names + tda_names

    # 2. define wrapper function
    def model_predict_wrapper(global_feats_matrix):
        # temp data for buffering
        temp_X = []
        for i in range(len(global_feats_matrix)):
            orig_graph = dataset.X[i % len(dataset)]
            new_graph = GraphData(
                node_features=orig_graph.node_features,
                edge_index=orig_graph.edge_index,
                edge_features=orig_graph.edge_features,
                global_features=global_feats_matrix[i].astype(np.float32)
            )
            temp_X.append(new_graph)
        
        temp_dataset = dc.data.NumpyDataset(X=np.array(temp_X, dtype=object))
        preds = model.predict(temp_dataset)
        # if y has been transformed, it will be returned to original value
        if transformers:
            preds = transformers[0].untransform(preds)
        return preds.flatten()

    # 3. KernelExplainer (for general models)
    # Background distribution
    background = X_global[np.random.choice(X_global.shape[0], 20, replace=False)]
    # Extract test data for explanation
    test_samples = X_global[np.random.choice(X_global.shape[0], 50, replace=False)]

    explainer = shap.KernelExplainer(model_predict_wrapper, background)
    shap_values = explainer.shap_values(test_samples)

    # 4. Visualization
    # Beeswarm Plot
    plt.figure(figsize=(5, 5),dpi=300)
    shap.summary_plot(shap_values, test_samples, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot (Global Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_summary.png"), dpi=300)
    plt.close()

    # Bar Plot
    plt.figure(figsize=(5, 5),dpi=300)
    shap.summary_plot(shap_values, test_samples, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_importance_bar.png"), dpi=300)
    plt.close()

# 2. data load and save
if os.path.exists(dataset_cache_dir):
    print(f"have data and load from {dataset_cache_dir}")
    # print(f"old cache is cleaning")
    # shutil.rmtree(dataset_cache_dir)
    dataset = dc.data.DiskDataset(dataset_cache_dir)
else:
    print("none data, start to read the original csv file")
    df_main = pd.read_csv(main_csv)
    featurizer = dc.feat.DMPNNFeaturizer()
    temp_graphs = []   # 存 Graph 结构
    temp_globals = []  # 存 [T, G, P, TDA(15)]
    y_list, w_list, f_ids = [], [], []

    for index, row in df_main.iterrows():
        try:
            fid = str(int(float(row['File ID'])))
        except:
            fid = str(row['File ID']).strip()
            
        target_csv = os.path.join(csv_folder, f"{fid}.csv")
        
        if os.path.exists(target_csv):
            mol = csv_to_rdkit_mol_safe(target_csv)
            if mol:
                feat = featurizer.featurize([mol])
                if len(feat) > 0:
                    T = float(row['Temperature (K)'])
                    P = float(row['Pressure (bar)'])
                    # G = float(row['G'])
                    inv_T = 1.0 / T 
                    log_P = np.log(P + 1e-6)
                    phys_feats = [T, P]
                    tda_fingerprint = compute_tda_ripser_15d(mol)
                    # phys_feats = [float(row['Temperature (K)']), float(row['G']), float(row['Pressure (bar)'])]
                    combined_global = np.concatenate([phys_feats, tda_fingerprint])
                    
                    temp_graphs.append(feat[0])
                    temp_globals.append(combined_global)
                    y_list.append([row['theta']])
                    w_list.append([1.0])
                    f_ids.append(fid)

        if (index + 1) % 10 == 0:
            print(f"process: have done {index + 1} files")
    temp_globals = np.array(temp_globals)
    scaler = StandardScaler()
    scaled_globals = scaler.fit_transform(temp_globals)
    X_list = []
    for i in range(len(temp_graphs)):
        new_feat = GraphData(
            node_features=temp_graphs[i].node_features,
            edge_index=temp_graphs[i].edge_index,
            edge_features=temp_graphs[i].edge_features,
            global_features=scaled_globals[i]
        )
        X_list.append(new_feat)
    # transfer to numpy and save
    X = np.array(X_list, dtype=object)
    y = np.array(y_list)
    w = np.array(w_list)
    dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, w=w, ids=f_ids, data_dir=dataset_cache_dir)
print(f"data have been saved in: {dataset_cache_dir}")

# 3. data splitting and preparation ---
print(f"valid data: {len(dataset)}")
splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8, seed=42)

# # split by file id
# all_ids = dataset.ids.flatten() 
# unique_fids = np.unique(all_ids)

# random.seed(42)
# random.shuffle(unique_fids)

# split_idx = int(len(unique_fids) * 0.8)
# train_fids_pool = set(unique_fids[:split_idx])
# train_indices = []
# test_indices = []
# for i in range(len(dataset)):
#     current_id = all_ids[i]
#     if current_id in train_fids_pool:
#         train_indices.append(i)
#     else:
#         test_indices.append(i)

# train_dataset = dataset.select(train_indices)
# test_dataset = dataset.select(test_indices)

# print(f"training set ids: {len(train_fids_pool)}, test set ids: {len(unique_fids)-len(train_fids_pool)}")

transformers = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
]

for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    test_dataset = transformer.transform(test_dataset)

# 4. model construction and training
model = dc.models.torch_models.DMPNNModel(
    mode='regression',
    n_tasks=1,
    batch_size=64,
    global_features_size=17,  #15TDA
    enc_hidden=32,  # Size of hidden layer in the encoder layer
    ffn_hidden=32,  # Size of hidden layer in the feed-forward network layer
    ffn_layers=3,    # Number of layers in the feed-forward network layer
    # learning_rate=1e-4,
    ffn_dropout_p = 0.3,
    learning_rate=dc.models.optimizers.ExponentialDecay(5e-4, 0.9, 1000)
)
# print(model.model)
print("start training")
num_epochs = 500
train_losses = []
for epoch in range(num_epochs):
    loss = model.fit(train_dataset, nb_epoch=1)
    train_losses.append(loss)
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.6f}")

# 5. evaluation and plot
metric = dc.metrics.Metric(dc.metrics.mae_score)
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print(f"Training MAE: {train_scores['mae_score']:.4f}")
print(f"Test MAE: {test_scores['mae_score']:.4f}")
y_pred_transformed = model.predict(test_dataset)
y_pred = transformers[0].untransform(y_pred_transformed)
y_true = transformers[0].untransform(test_dataset.y)
SAVE_DIR = '/home/ucaqzti/deepchem'
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()
r2 = r2_score(y_true_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, edgecolors='k', c='#1f77b4', label='Predictions')
line_min = min(y_true_flat.min(), y_pred_flat.min())
line_max = max(y_true_flat.max(), y_pred_flat.max())
plt.plot([line_min, line_max], [line_min, line_max], 'r--', lw=2, label='Ideal')
stats_text = f"$R^2 = {r2:.3f}$\n$RMSE = {rmse:.4f}$"
plt.gca().text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
plt.xlabel('Actual Theta', fontsize=12)
plt.ylabel('Predicted Theta', fontsize=12)
plt.title('DMPNN Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "result.png")
plt.savefig(save_path)
plot_learning_curve(train_losses)
shap_analysis(model, test_dataset, transformers)