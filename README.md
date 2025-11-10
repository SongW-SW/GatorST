

# GatorST: A Versatile Contrastive Meta-Learning Framework for Spatial Transcriptomic Data Analysis


## Requirements
- python : 3.9.12
- scanpy : 1.10.3
- sklearn : 1.1.1
- scipy : 1.9.0
- torch : 1.11.2
- torch-geometric : 2.1.0
- numpy : 1.24.4
- pandas : 2.2.3


## Project Structure

```bash
.
├── main.py            # Main training and evaluation loop
├── model.py           # Model architecture and loss functions
├── data_loader.py     # Data loading and graph construction utilities
├── util.py            # Utility functions (seed setup, metrics, dropout)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
├── saved_graph/       # Folder for cached graphs and subgraphs
└── result.json        # Evaluation results output
```

## Usage

### **1. Prepare your input data**

Place your **.h5ad** spatial transcriptomics datasets in the `./data/` directory.
Each `.h5ad` file should contain:

* `adata.X`: Gene expression matrix
* `adata.obs`: Spot/cell-level metadata
* `adata.obsm["spatial"]`: Spatial coordinates

Example:

```bash
data/
 ├── 15_processed_all.h5ad
 ├── mouse_brain_section1.h5ad
 └── breast_cancer_blockA.h5ad
```

---

### **2. Run training and evaluation**

Execute the main script to train and evaluate across datasets:

```bash
python main.py
```

#### Optional configuration inside `main.py`:

* `epochs`: number of training epochs per run (default 50)
* `batch_size`: number of samples per batch (default 20)
* `lr`: learning rate (default 0.001)
* `runs`: repeated random seed experiments (default 10)

You can modify these directly in `main.py`:

```python
epochs = 50
batch_size = 20
lr = 0.001
```

To run only selected datasets, edit the filtering condition:

```python
if data_name not in ['15_processed_all']:
    continue
```

---

### **3. Output files**

After training completes:

* Trained model checkpoints: `saved_models/`

  ```
  saved_models/
   ├── 15_processed_all_model_run_0.pt
   ├── 15_processed_all_model_run_1.pt
   ...
  ```
* Evaluation results (accuracy, clustering, etc.): `result.json`

  ```json
  {
      "15_processed_all": [
          {"ARI": 0.82, "NMI": 0.79, "Silhouette": 0.67},
          ...
      ]
  }
  ```
* Intermediate subgraph structures (optional): `saved_graph/`

---

### **4. Run-time behavior**

When executed, the program automatically:

1. Iterates through each `.h5ad` file in `./data/`
2. Builds train/validation/test DataLoaders via `loader_construction()`
3. Runs `train()` for multiple seeds and saves the best checkpoint per run
4. Evaluates the model using `test()`
5. Logs all metrics to `result.json`

A typical terminal output:

```
Start Running 15_processed_all
[Run 0] Training Epoch 50 | Best Epoch: 37 | Min Loss: 0.0123
[Run 0] Test ARI: 0.812 | NMI: 0.784
Results saved to result.json
```
## Outputs
- Trained models saved in `saved_models/`
- Intermediate subgraph representations saved in `saved_graph/`
- Final clustering results and metrics saved in `result.json`

## Datasets
- LIBD Human Dorsolateral Prefrontal Cortex (DL: http://research.libd.org/spatialLIBD/
- Human Breast Cancer: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0
- Mouse Brain Tissue: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0



## Citation

```
@article{wang2025gatorst,
  title={GatorST: A Versatile Contrastive Meta-Learning Framework for Spatial Transcriptomic Data Analysis},
  author={Wang, Song and Liu, Yuxi and Zhang, Zhenhao and Ma, Qin and Song, Qianqian and Bian, Jiang},
  journal={bioRxiv},
  year={2025}
}
```
