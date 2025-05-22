# Learning Survival Distributions with the Asymmetric Laplace Distribution (ICML 2025)

This repository contains the official implementation of our ICML 2025 paper:

**Learning Survival Distributions with the Asymmetric Laplace Distribution**  
📄 [Paper on arXiv](https://arxiv.org/abs/2505.03712)

## 🔍 Overview
We propose a parametric survival analysis framework based on the Asymmetric Laplace Distribution (ALD), which enables closed-form computation of key distributional summaries such as mean, median, mode, and quantiles. Our method learns individual-level ALD parameters (location, scale, and asymmetry) via maximum likelihood estimation, supporting both observed and censored data. Compared to traditional and neural baselines—including DeepSurv, DeepHit, and CQRNN—our model achieves superior performance in terms of accuracy, calibration, and discrimination across 21 datasets and 9 evaluation metrics. This approach provides a robust and interpretable alternative for survival modeling in both synthetic and real-world applications.

## 🗂️ Directory Structure

```
datasets/
├── breast_msk_2018_clinical_data.tsv   
├── gbsg_cancer_train_test.h5            
├── gbsg.csv                             
├── lgggbm_tcga_pub_clinical_data.tsv    
├── metabric_IHC4_clinical_train_test.h5  
├── support_train_test.h5                
├── tmb_immuno_mskcc.tsv                 
└── whas_train_test.h5                  # real-world data with real censoring datasets

figures/
├── *.png                               # Figures and visualizations generated during analysis

res/
├── *.json                              # Experiment results

./
├── datasets.py                         # Functions for loading and preprocessing datasets
├── hyperparams.py                      # Hyperparameters used in the experiments 
├── models.py                           # Neural network architectures and custom losses in the experiments 
├── script.py                           # Main script for running experiments
├── utils.py                            # Utility functions for common operations

requirements.txt                        # Python dependencies
README.md                               # Project description and usage instructions
```

---

## **Requirements**
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Data Preparation**
Place all datasets in the `datasets/` folder. Preprocessing functions are available in `datasets.py`.

### **2. Running Experiments**
Use `script.py` to train and evaluate models:
```bash
python script.py
```

### **3. Visualizations**
Figures and plots generated during analysis will be saved in the `figures/` folder.
