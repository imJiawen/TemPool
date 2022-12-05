# TemPool

## Overview

This repository contains the implementation of a hierarchical pooling approach for the temporal graph. This repository is structured as follows:

- **./model/**: 
    - ```tempool.py```: TemPool module implementation.
    - ```poolastgcn.py```:  backbone model ASTGCN equipped with the TemPool pooling layer.
- **./util/**: 
    - ```util.py```: utility scripts.
- **run.py**: running script.
- **run.sh**: executable shell script.

## Requirements

This work is implemented based on [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) library.

Your local system should have the following executables:

- Python 3.7 or later
- git

All required libraries are listed in [requirement.txt](./requirement.txt) and can be installed with

```bash
pip install -r requirement.txt
```

## Datasets

To validate the effectiveness of Tempool, we select five datasets for each type of temporal graph. More details can be found at [PyTorch Geometric Temporal Dataset](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html).

   |  Dataset   | Abbr.  | Signal | Graph | Frequency | \|V\| |
   |  :----  | :----: | :----: | :----: | :----: | :----: |
   | Chickenpox Hungary | CKP    | Temporal  | Static | Weekly  | 20 |
   | Montevideo Bus     | BUS    | Temporal  | Static | Hourly  | 675 |
   | Twitter Tennis RG  | TWI-RG | Static    | Dynamic | Hourly | 1000 |
   | Twitter Tennis UO  | TWI-UO | Static    | Dynamic | Hourly | 1000 |
   | England Covid-19   | EnCOV  | Temporal  | Dynamic | Daily  | 129 |

## Quick Start

Please execute the Shell script to run the backbone model with the proposed pooling layers:

```bash
bash run.sh
```

## Run the model

Arguments used for running our model and baselines:

```bash
python run.py --model_name {model_name} \
                --dataset {dataset} --epoch_num {epoch_num} \
                --g_ratio {g_ratio} --t_ratio {t_ratio}
```

   - ```model_name```: the model name, select from ```['astgcn','a3tgcn2','agcrn','dcrnn','stconv','mstgcn','tgcn','evolgcn','lrgcn','mpnnlstm', 'poolastgcn']```.
   - ```dataset```: the dataset name, select from ```['ckp', 'eng','twi_rg17', 'twi_uo17','bus']```.
   - ```epoch_num```: training epoch number.
   - ```g_ratio```: control the cluster number of nodes, used for the ablation study.
   - ```t_ratio```: control the segment number of snapshots, used for the ablation study.
   
   For more details, please refer to ```run.py```.


## Experimental Results

The Mean Square Error (MSE) results over five temporal graph datasets.

|  Model   | CKP  | BUS | TWI-RG | TWI-UO | EnCOV |
|  :----  | :----: | :----: | :----: | :----: | :----: |
| [STGCN](https://arxiv.org/abs/1709.04875)                     | 1.053 | 1.041 | 0.451 | 0.577 |  1.446 |
| [AGCRN](https://arxiv.org/abs/2007.02842)                     | 1.093 | 1.134 | 0.670 | 0.882 |  1.370 |
| [T-GCN](https://arxiv.org/abs/1811.05320)                     | 1.045 | 1.009 | 0.453 | 0.610 |  0.690 |
| [A3T-GCN](https://arxiv.org/abs/2006.11583)                   | 1.039 | 1.005 | 0.447 | 0.592 |  0.768 |
| [EvolveGCN](https://arxiv.org/abs/1902.10191)                 | 1.089 | 1.482 | 0.424 | 0.549 |  0.552 |
| [DCRNN](https://arxiv.org/abs/1707.01926)                     | 1.057 | 0.980 | 0.419 | 0.540 |  0.943 |
| [LRGCN](https://arxiv.org/abs/1905.03994)                     | 1.031 | 0.975 | 0.424 | 0.544 |  0.976 |
| [MPNN LSTM](https://arxiv.org/abs/2009.08388)                 | 1.074 | 1.021 | 0.422 | 0.538 |  1.129 |
| [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881)| 1.123 | 1.165 | 0.485 | 0.606 | 1.200 |
| (+TemPool)             | 0.9851 | 0.975 | 0.425 | 0.542 | 0.508 |


## License

Our code in this repository is licensed under the [MIT license](https://github.com/youngKG/Multi-Granularity-ClinTS/blob/main/LICENSE).

