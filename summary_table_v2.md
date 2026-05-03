# Summarized comparison table
## 1. Conventional machine learning models
### 1.1 Feature construction
- phys: 
  `length,n,m,diameter,Ti_count,FG_C=O,FG_NH2,FG_SO3H,FG_none,TiType_1Ti_substitution,TiType_1Ti_surface,TiType_2Ti_substitution,TiType_2Ti_surface,TiType_none`
- T,P: 
  `Temperature,Pressure`
- TDA: 
  `TDA_H0_max,TDA_H0_min,TDA_H0_mean,TDA_H0_std,TDA_H0_sum,TDA_H1_max,TDA_H1_min,TDA_H1_mean,TDA_H1_std,TDA_H1_sum,TDA_H2_max,TDA_H2_min,TDA_H2_mean,TDA_H2_std,TDA_H2_sum`
- **target**: 
  `theta`

### 1.2 Summary table
| Model | phys_T,P_TDA | phys_TDA | phys_T,P | only_phys | only_TDA | only_T,P |
| --- | --- | --- | --- | --- | --- | --- |
| Linger Regression | 0.712 | 0.642 | 0.437 | 0.249 | 0.195 | 0.107 |
| XGBoost | 0.998 | 0.837 | 0.871 | 0.724 | 0.698 | 0.109 |
| SVR | 0.998 | 0.825 | 0.804 |  0.712 | 0.677 | 0.101 |
| Random Forest | 0.994 | 0.838 | 0.818 | 0.726 | 0.700 | 0.105 |

## 2. Graph neural networks
### 2.1 Feature construction
- graph:
  `The unoptimised coordinates of the complex (SWCNT+H2)`
- TDA:
  `TDA_H0_max,TDA_H0_min,TDA_H0_mean,TDA_H0_std,TDA_H0_sum,TDA_H1_max,TDA_H1_min,TDA_H1_mean,TDA_H1_std,TDA_H1_sum,TDA_H2_max,TDA_H2_min,TDA_H2_mean,TDA_H2_std,TDA_H2_sum` 
- T,P:
  `Temperature,Pressure`
### 2.2 Summary table
| Model | graph+T,P+TDA | graph+T,P | graph+TDA | only graph (44*) | only graph (1500*) |  only T,P (MLP) | only TDA (MLP) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Deepchem GNN | 0.907 | 0.880 | 0.815 | 0.233 | 0.850 | 0.131 | 0.539 |

*44: the 44 original SWCNT structures with H2
*1500 the augmented 1500 structures with perturbation on the the position of H2
