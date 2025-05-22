## Learning Survival Distributions with the Asymmetric Laplace Distribution

## **Directory Structure**
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
