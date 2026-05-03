# NSCI0017 Project

## Guide of this repository
```python
|---0430summary
   |---conventional_ml # RF,SVR,LR and XGBoost
       |---MLs.py # the main code for running these models
       |---requirement.txt
       |---dataset_full_feat.csv # the dataset with all the features for the 4 models
       |...5 folders of output

   |---dc_gnn # the deepchem GNN model
       |---graph_T,P
           |---data_prepare # the structures of input data for GNN
           |---dataset_form.csv
           |---dc_gnn_graph_tp.py
           |---figures.png # the results of the GNN model with graph+T,P features
       |---graph_T,P_TDA
       |---graph_TDA  
       |---graph(44)
       |---graph(1500)
       |---T,P
       |---TDA
       |---dcgnn_requirement.txt      
```


## Conventional ML models 
### Conda evnironment setup
All the packages used in this project are listed in the `requirements.txt` file. You can create a new conda environment and install the required packages using the following commands:
```bash
pip install -r requirements.txt
```
### Running the code
There are 4 models, including Random Forest, Support Vector Regression, XGBoost and Linear Regression. And there are 5 modes for the models with different input features:
- phys: 
  ```
  length,n,m,diameter,Ti_count,FG_C=O,FG_NH2,FG_SO3H,FG_none,TiType_1Ti_substitution,TiType_1Ti_surface,TiType_2Ti_substitution,TiType_2Ti_surface,TiType_none
  ```
- T,P: 
  ```
  Temperature,Pressure
  ```
- TDA: 
  ```
  TDA_H0_max,TDA_H0_min,TDA_H0_mean,TDA_H0_std,TDA_H0_sum,TDA_H1_max,TDA_H1_min,TDA_H1_mean,TDA_H1_std,TDA_H1_sum,TDA_H2_max,TDA_H2_min,TDA_H2_mean,TDA_H2_std,TDA_H2_sum
  ```
- Target: 
  ```
  theta
  ```
To run the code, you can change the `ANALYSIS_MODE` in the `MLs.py` file to run the 4 models with different input features.

```python
# ============ CONFIGURATION ============
# choose the type: "phys_T,P_TDA", "phys", "T,P", "TDA", "phys_T,P", "phys_TDA"
ANALYSIS_MODE = "phys"
```

The code will use the `dataset_full_feat.csv` as the input data, and drop some parts of the columns based on the setting of `ANALYSIS_MODE`. The results will be saved in the corresponding output directory, such as `phys`, `T,P`, `TDA`, etc. You can find the results in the output directory after running the code.

## Deepchem GNN model 
### Conda evnironment setup
All the packages used in this project are listed in the `dcgnn_requirement.txt` file. You can create a new conda environment and install the required packages using the following commands:
```bash
pip install -r dcgnn_requirement.txt
```
### Running the code
For each subfolder in the `dc_gnn` folder, there is a corresponding code for running the GNN model with different input features. For example, to run the GNN model with graph+T,P features, you can run the `dc_gnn_graph_tp.py` file. The input data for the GNN model is prepared in the `data_prepare` folder, and the results will be saved in the same folder after running the code. You can find the results in the `*.png` files after running the code.