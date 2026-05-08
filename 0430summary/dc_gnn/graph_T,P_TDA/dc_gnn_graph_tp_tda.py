import os
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
import warnings
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
# import shutil
import random
from deepchem.feat.graph_data import GraphData
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import shap
from scipy.stats import norm
from sklearn.utils import resample
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DGLBACKEND'] = 'pytorch'
# 1. path setting
csv_folder = '/home/ucaqzti/deepchem/data_prepare'
main_csv = '/home/ucaqzti/models/dataset_form.csv'
dataset_cache_dir = './dmpnn_dataset_cache'  # cache folder


import torch
import numpy as np
from torch_geometric.data import Batch

def analyze_ig_statistics(model, dataset, csv_folder, save_dir):
    """
    statistical analysis of IG importance scores across the test set, and visualize the distribution of importance with box plots
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print("\n=== IG Statistical Analysis (Table-based) ===")

    # ===============================
    # IG importance mapping
    # ===============================
    # 1. define the mapping from file ID to structure attributes
    structure_map = {
        1: ("6,6", 10, "none", "none"),
        2: ("6,6", 13, "none", "none"),
        3: ("6,6", 15, "none", "none"),
        4: ("6,6", 20, "none", "none"),
        5: ("6,6", 10, "SO3H", "none"),
        6: ("6,6", 10, "C=O", "none"),
        7: ("6,6", 10, "NH2", "none"),
        8: ("6,6", 10, "none", "1Ti_surface"),
        9: ("6,6", 10, "none", "1Ti_substitution"),
        10: ("6,6", 10, "none", "2Ti_surface"),
        11: ("6,6", 10, "none", "2Ti_substitution"),

        12: ("7,5", 10, "none", "none"),
        13: ("7,5", 13, "none", "none"),
        14: ("7,5", 15, "none", "none"),
        15: ("7,5", 20, "none", "none"),
        16: ("7,5", 10, "SO3H", "none"),
        17: ("7,5", 10, "C=O", "none"),
        18: ("7,5", 10, "NH2", "none"),
        19: ("7,5", 10, "none", "1Ti_surface"),
        20: ("7,5", 10, "none", "1Ti_substitution"),
        21: ("7,5", 10, "none", "2Ti_surface"),
        22: ("7,5", 10, "none", "2Ti_substitution"),

        23: ("13,1", 10, "none", "none"),
        24: ("13,1", 13, "none", "none"),
        25: ("13,1", 15, "none", "none"),
        26: ("13,1", 20, "none", "none"),
        27: ("13,1", 10, "SO3H", "none"),
        28: ("13,1", 10, "C=O", "none"),
        29: ("13,1", 10, "NH2", "none"),
        30: ("13,1", 10, "none", "1Ti_surface"),
        31: ("13,1", 10, "none", "1Ti_substitution"),
        32: ("13,1", 10, "none", "2Ti_surface"),
        33: ("13,1", 10, "none", "2Ti_substitution"),

        34: ("13,2", 10, "none", "none"),
        35: ("13,2", 13, "none", "none"),
        36: ("13,2", 15, "none", "none"),
        37: ("13,2", 20, "none", "none"),
        38: ("13,2", 10, "SO3H", "none"),
        39: ("13,2", 10, "C=O", "none"),
        40: ("13,2", 10, "NH2", "none"),
        41: ("13,2", 10, "none", "1Ti_surface"),
        42: ("13,2", 10, "none", "1Ti_substitution"),
        43: ("13,2", 10, "none", "2Ti_surface"),
        44: ("13,2", 10, "none", "2Ti_substitution"),
    }

    # ===============================
    # 2. calculation and record the IG importance
    # ===============================
    records = []
    test_samples = dataset.itersamples()

    for i, (x_obj, y_val, w_val, id_val) in enumerate(test_samples):
        fid = str(id_val)

        try:
            idx = int(fid)

            if idx not in structure_map:
                print(f"Warning: {fid} not in mapping, skipped")
                continue

            chirality, length, func, doping = structure_map[idx]

            scores = explain_with_ig_final(model, x_obj, n_steps=30)
            scores = np.abs(scores)

            if np.max(scores) > 1e-8:
                scores = scores / np.max(scores)

            mean_score = np.mean(scores)
            max_score = np.max(scores)
            std_score = np.std(scores)

            k = max(1, int(0.1 * len(scores)))
            top_mean = np.mean(np.sort(scores)[-k:])

            records.append({
                "fid": fid,
                "chirality": chirality,
                "length": length,
                "functional_group": func,
                "doping": doping,
                "mean_ig": mean_score,
                "max_ig": max_score,
                "std_ig": std_score,
                "top10_ig": top_mean
            })

        except Exception as e:
            print(f"Error processing {fid}: {e}")

    df = pd.DataFrame(records)

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "ig_statistics.csv"), index=False)

    print("✓ IG statistics saved")

    # ===============================
    # 3. box plot visualization
    # ===============================
    def boxplot(feature, metric, filename):
        plt.figure(figsize=(5,5), dpi=300)
        groups = df.groupby(feature)[metric].apply(list)

        plt.boxplot(groups.values, labels=groups.index)
        plt.xlabel(feature)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {feature}")

        path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(f"✓ Saved: {filename}")

    boxplot("chirality", "mean_ig", "ig_chirality_mean.png")
    boxplot("length", "mean_ig", "ig_length_mean.png")
    boxplot("functional_group", "top10_ig", "ig_functional_top10.png")
    boxplot("doping", "top10_ig", "ig_doping_top10.png")
    boxplot("doping", "std_ig", "ig_doping_std.png")


    print("=== IG analysis DONE ===")

def save_3d_importance_pdb(fid, scores, csv_folder, output_pdb_path):
    """
    IG score -> original xyz -> pdb files
    B-factor -> importance score
    """
    # 1. read coordinates (eg. 1.csv)
    csv_path = os.path.join(csv_folder, f"{fid}.csv")
    df = pd.read_csv(csv_path)
    coords = df[['X', 'Y', 'Z']].values
    elements = df['Element'].values

    # 2. Normalization (0-100)
    max_s = np.abs(scores).max()
    norm_scores = (np.abs(scores) / max_s) * 100 if max_s > 0 else scores

    # 3. write to pdb files
    with open(output_pdb_path, 'w') as f:
        for i, (pos, el, score) in enumerate(zip(coords, elements, norm_scores)):
            f.write(f"ATOM  {i+1:>5} {el:<4} MOL A   1    {pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}  1.00{score:>6.2f}\n")
        f.write("END\n")
    print(f"3D importance has been saved in: {output_pdb_path}")

def explain_with_ig_final(dc_model, graph_obj, n_steps=50):
    py_model = dc_model.model
    py_model.eval()
    
    # 1. extract the initial pyg_batch object
    intercepted = {}
    def hook(module, input):
        intercepted['batch'] = input[0]
        return None
    
    # DMPNNModel: forward input
    h = py_model.register_forward_pre_hook(hook)
    try:
        _ = dc_model.predict_on_batch([graph_obj])
    finally:
        h.remove()
        
    pyg_batch = intercepted['batch']
    
    # 2. Find feature terms
    # find the feature f_ini_atoms_bonds in DPMNN
    f_ini_orig = pyg_batch['f_ini_atoms_bonds'].clone().detach().to(torch.float32)
    f_ini_baseline = torch.zeros_like(f_ini_orig)
    total_grads = torch.zeros_like(f_ini_orig)

    print(f"IG integral (50 steps)...")
    for alpha in np.linspace(0.1, 1.0, n_steps):
        f_step = f_ini_baseline + alpha * (f_ini_orig - f_ini_baseline)
        f_step.requires_grad = True
        
        # core: integrate the graident with Tensor into pyg_batch
        pyg_batch['f_ini_atoms_bonds'] = f_step
        
        with torch.set_grad_enabled(True):
            # through py_model to run, without numpy transformation of DeepChem
            output = py_model(pyg_batch) 
            # the output is torch.Tensor with grad_fn
            loss = output.sum()
            py_model.zero_grad()
            loss.backward()
            
            if f_step.grad is not None:
                total_grads += f_step.grad

    # Mapping and normalization
    # integral calculation (Input - Baseline) * Avg_Gradients
    # the dimension of result is (E,)，means the normalized score of each edges
    edge_importance = ((f_ini_orig - f_ini_baseline) * (total_grads / n_steps)).sum(dim=-1).detach().cpu().numpy()
    
    num_nodes = graph_obj.node_features.shape[0]
    atom_scores = np.zeros(num_nodes)
    edge_index = graph_obj.edge_index # generally is (2, E)
    
    # Accumulate the score onto Source Node
    for i in range(min(len(edge_index[0]), len(edge_importance))):
        src_node = int(edge_index[0][i])
        # get absolute value
        atom_scores[src_node] += np.abs(edge_importance[i])
        
    # avoid all 0 or 1 score
    score_max = atom_scores.max()
    
    if score_max > 1e-9:
        atom_scores /= score_max
        print(f"DEBUG: IG successfully calculated, and the max score (initial): {score_max:.6f}")
    else:
        # if score_max ~ 0, there is no valid gradient has been captured
        print("Warning: all the atoms get 0 score! May result from all gradient elimination or the model has no sensitivity with the structures")
        # keep the original format without division
        atom_scores = np.zeros(num_nodes)

    return atom_scores

def visualize_atom_importance_final(fid, atom_scores, csv_folder, save_path):
    from rdkit.Chem.Draw import rdMolDraw2D
    from matplotlib import cm
    import matplotlib.colors as clr
    from rdkit.Chem import rdDepictor
    print(f"Generating clear, large importance map for: {fid}...")
    
    target_csv = os.path.join(csv_folder, f"{fid}.csv")
    mol = csv_to_rdkit_mol_safe(target_csv)
    
    if mol is None:
        print(f"Could not load molecule {fid} for visualization.")
        return

    rdDepictor.Compute2DCoords(mol)
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms != len(atom_scores):
        print(f"Warning: Atom count mismatch! {num_atoms} vs {len(atom_scores)}. Fixing...")
        if len(atom_scores) > num_atoms:
            atom_scores = atom_scores[:num_atoms]
        else:
            atom_scores = np.pad(atom_scores, (0, num_atoms - len(atom_scores)))

    # 2. set the figure size
    width, height = 4000, 2000
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height) # PNG
    except:
        # if Cairo error, change to use SVG
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        print(" Cairo error, falling back to SVG (image_xxx.svg).")

    # 3. mapping the color
    
    norm = clr.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Reds') # red color series
    
    atom_colors = {}
    for i, score in enumerate(atom_scores):
        rgb = cmap(norm(float(score)))[:3] # extract RGB
        atom_colors[i] = rgb

    # 4. visualisation
    drawer.DrawMolecule(mol, highlightAtoms=list(range(num_atoms)), highlightAtomColors=atom_colors)
    
    # 5. save
    drawer.FinishDrawing()
    with open(save_path, 'wb') as f:
        f.write(drawer.GetDrawingText())
        
    print(f"Final clear importance map saved to: {save_path}")

from scipy.stats import norm

def calibrate_uncertainty_scale(y_true, mean_pred, std_pred):
    """
    Find optimal scaling factor to calibrate uncertainty
    """

    scales = np.linspace(0.5, 5.0, 50)  # search range for scaling factor
    best_scale = 1.0
    best_error = 1e10

    significance_levels = np.arange(0.1, 1.0, 0.1)

    for scale in scales:
        std_scaled = std_pred * scale
        error_total = 0
        
        for alpha in significance_levels:
            z = norm.ppf(1 - alpha / 2)
            abs_error = np.abs(y_true - mean_pred)
            out_of_bounds = abs_error > (z * std_scaled)
            error_rate = np.mean(out_of_bounds)

            error_total += (error_rate - alpha)**2

        if error_total < best_error:
            best_error = error_total
            best_scale = scale

    print(f"Optimal calibration scale: {best_scale:.3f}")
    return best_scale

def plot_overfitting_check_gnn(all_preds, y_true, save_dir):

    plt.figure(figsize=(6,5), dpi=300)

    r2_scores = []

    for i in range(all_preds.shape[0]):
        pred = all_preds[i]
        r2 = r2_score(y_true, pred)
        r2_scores.append(r2)

    r2_scores = np.array(r2_scores)

    mean = np.mean(r2_scores)
    std = np.std(r2_scores)

    plt.hist(r2_scores, bins=10, alpha=0.6, color='steelblue')

    plt.axvline(mean, linestyle='--', color='red',
                label=f"Mean={mean:.3f}, Std={std:.3f}")

    plt.xlabel("R² (Bootstrap Models)")
    plt.ylabel("Frequency")
    plt.title("Overfitting Check via Bootstrap (GNN)")
    plt.legend()

    save_path = os.path.join(save_dir, "gnn_overfitting_check.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✓ Saved: {save_path}")

def plot_error_rate_vs_significance_gnn(y_true, mean_pred, std_pred, save_dir):

    plt.figure(figsize=(6,5), dpi=300)

    significance_levels = np.arange(0.05, 1.05, 0.05)
    error_rates = []

    abs_error = np.abs(y_true - mean_pred)

    for alpha in significance_levels:
        z = norm.ppf(1 - alpha / 2)
        out_of_bound = abs_error > (z * std_pred)
        error_rate = np.mean(out_of_bound)
        error_rates.append(error_rate)

    plt.plot(significance_levels, error_rates, 'o-', label='DMPNN')

    # ideal line
    plt.plot([0,1],[0,1],'k--',label='Ideal')

    plt.xlabel("Significance Level")
    plt.ylabel("Error Rate")
    plt.title("Uncertainty Calibration (GNN)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, "gnn_error_rate_vs_significance.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✓ Saved: {save_path}")


def bootstrap_uncertainty(train_dataset, test_dataset, transformer, n_models=5):
    """
    Bootstrap uncertainty for DeepChem DMPNN
    """

    all_preds = []

    for i in range(n_models):
        print(f"Bootstrap model {i+1}/{n_models}")

        # 1. bootstrap resample
        indices = np.random.choice(len(train_dataset.y), size=len(train_dataset.y), replace=True)

        boot_dataset = dc.data.NumpyDataset(
            X=train_dataset.X[indices],
            y=train_dataset.y[indices],
            ids=[train_dataset.ids[j] for j in indices]
        )
        boot_dataset = transformer.transform(boot_dataset)
        # 2. construct model
        model = dc.models.torch_models.DMPNNModel(
            mode='regression',
            n_tasks=1,
            batch_size=64,
            global_features_size=17,
            enc_hidden=32,
            ffn_hidden=32,
            ffn_layers=3,
            ffn_dropout_p=0.3,
            learning_rate=dc.models.optimizers.ExponentialDecay(5e-4, 0.9, 1000)
        )

        # 3. train model
        model.fit(boot_dataset, nb_epoch=150)

        # 4. predict
        pred = model.predict(test_dataset)

        # 5. untransform
        pred = transformer.untransform(pred)

        all_preds.append(pred.flatten())

    all_preds = np.array(all_preds)

    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)

    return mean_pred, std_pred, all_preds


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
    phys_names = ['Temperature (K)', 'Pressure (bar)']
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
train_dataset_raw = train_dataset
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

print("\nRunning bootstrap uncertainty...")

mean_pred, std_pred, all_preds = bootstrap_uncertainty(
    train_dataset=train_dataset_raw,
    test_dataset=test_dataset,
    transformer=transformers[0],
    n_models=20
)


# calibration
scale = calibrate_uncertainty_scale(
    y_true_flat,
    mean_pred,
    std_pred
)

std_pred_calibrated = std_pred * scale


plot_error_rate_vs_significance_gnn(
    y_true_flat,
    mean_pred,
    std_pred_calibrated,
    SAVE_DIR
)


plot_overfitting_check_gnn(
    all_preds,
    y_true_flat,
    SAVE_DIR
)



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
plt.figure(figsize=(6,6), dpi=300)

plt.errorbar(
    y_true_flat,
    mean_pred,
    yerr=1.96 * std_pred_calibrated,
    fmt='o',
    alpha=0.5,
    ecolor='gray',
    capsize=2
)

plt.plot([y_true_flat.min(), y_true_flat.max()],
         [y_true_flat.min(), y_true_flat.max()],
         'r--')

plt.xlabel("Measured")
plt.ylabel("Predicted ± 95% CI")
plt.title("Prediction Intervals (DMPNN)")

plt.tight_layout()
save_dir = os.path.join(SAVE_DIR, "gnn_prediction_interval.png")
plt.savefig(save_dir)
plt.close()

plot_learning_curve(train_losses)
# Explainer (IG)
# ==============================
print("\n" + "="*30)
print("Start creating IG figures...")
print("="*30)

# 1. create the ig images folder
IG_SAVE_DIR = os.path.join(SAVE_DIR, 'test_set_ig_maps')
if not os.path.exists(IG_SAVE_DIR):
    os.makedirs(IG_SAVE_DIR)
    print(f"The folder was created {IG_SAVE_DIR}")

# 2. iteration
test_samples = test_dataset.itersamples()
total_test = len(test_dataset)
success_count = 0

print(f"find {total_test} samples and start analysing...")

# 3. for all samples
for i, (x_obj, y_val, w_val, id_val) in enumerate(test_samples):
    fid = str(id_val)
    print(f"[{i+1}/{total_test}] analysing ID: {fid} ...", end='\r')
    
    try:
        # 4. calculate the IG score
        scores = explain_with_ig_final(model, x_obj, n_steps=30)
        scores = np.abs(scores) # get absolute value for better visualization
        
        if np.max(scores) > 1e-8:
            scores = scores / np.max(scores)

        
        # 5. save the 2D png 
        vis_save_path = os.path.join(IG_SAVE_DIR, f"atom_ig_{fid}.png")
        visualize_atom_importance_final(fid, scores, csv_folder, vis_save_path)
        
        # 6. save the 3D pdb
        pdb_save_path = os.path.join(IG_SAVE_DIR, f"atom_ig_{fid}.pdb")
        save_3d_importance_pdb(fid, scores, csv_folder, pdb_save_path)
        
        success_count += 1
    except Exception as e:
        print(f"\nError processing ID {fid}: {e}")
analyze_ig_statistics(model, test_dataset, csv_folder, SAVE_DIR)
shap_analysis(model, test_dataset, transformers)