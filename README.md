# Learning Survival Distributions with the Asymmetric Laplace Distribution (ICML 2025)

This repository contains the official implementation of our ICML 2025 paper:

**Learning Survival Distributions with the Asymmetric Laplace Distribution**  📄 [Paper on arXiv](https://arxiv.org/abs/2505.03712)

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

## **Running Experiments**
Use `script.py` to train and evaluate models:
```bash
python script.py
```

---

## 📊 Datasets

We evaluate our method on two types of datasets:

### 🔬 Synthetic Datasets
- **Norm Linear / Non-linear**
- **Exponential**
- **Weibull**
- **LogNorm**
- **Norm-heavy / medium / light**
- **LogNorm-heavy / medium / light / same**

### 🧪 Real-World Datasets
- **METABRIC**: 9 features, 1523 train / 381 test, censoring: 0.42  
- **WHAS**: 6 features, 1310 train / 328 test, censoring: 0.57  
- **SUPPORT**: 14 features, 7098 train / 1775 test, censoring: 0.32  
- **GBSG**: 7 features, 1785 train / 447 test, censoring: 0.42  
- **TMBImmuno**: 3 features, 1328 train / 332 test, censoring: 0.49  
- **BreastMSK**: 5 features, 1467 train / 367 test, censoring: 0.77  
- **LGGGBM**: 5 features, 510 train / 128 test, censoring: 0.60  

Data sources include the [DeepSurv Repository](https://github.com/jaredleekatzman/DeepSurv) and [cBioPortal](https://www.cbioportal.org/).

---

## 📏 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Integrated Brier Score (IBS)** 
- **Harrell’s C-Index** 
- **Uno’s C-Index** 
- **Censored D-Calibration (CensDcal)** 
- **Cal[S(t|x)] (Slope / Intercept)**  
- **Cal[f(t|x)] (Slope / Intercept)**

---

## 🧠 Models

- **LogNorm MLE** – Parametric baseline using log-normal distribution and maximum likelihood estimation  
- **DSM (LogNorm / Weibull)** – Neural parametric mixture models with log-normal or Weibull components  
- **DeepSurv** – Semi-parametric Cox-based neural model for nonlinear covariate effects  
- **RSF** – Random Survival Forests  
- **GBM** – Gradient Boosted Models adapted for survival  
- **DeepHit** – Non-parametric neural model predicting discrete-time survival distributions  
- **CQRNN** – Censored quantile regression using Asymmetric Laplace loss  
- **ALD (proposed)** – Our method based on Asymmetric Laplace Distribution

---

## 📎 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{sheng2025learning,
  title={Learning Survival Distributions with the Asymmetric Laplace Distribution},
  author={Sheng, Deming and Henao, Ricardo},
  booktitle={International Conference on Machine Learning},
  pages={54772--54809},
  year={2025},
  organization={PMLR}
}
 ```
