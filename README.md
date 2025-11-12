# 🧠 Bank Customer Churn Prediction Framework 📊  
**Final Project — Data Analytics II (Pontificia Universidad Javeriana)**  

This project develops a complete **Machine Learning framework** to predict customer churn in a retail bank.  
It integrates data preprocessing, model training, hyperparameter tuning, and interpretability analysis using multiple algorithms (SVM, Neural Networks, and XGBoost).  
The goal is to identify clients most likely to leave the bank and understand the behavioral and financial factors driving churn.

---

## 🧩 Project Overview

A financial institution experienced a **15% client reduction in the last quarter**, prompting an analysis to predict which customers are likely to churn.  
This notebook creates a modular and explainable modeling pipeline to classify customers into churn (1) or retain (0), allowing the company to take preventive actions.

---

## 🗂️ Repository Structure

ChurnPrediction_ML_Framework/
│
├── Notebook_Main.ipynb # Main notebook with full framework pipeline
│
├── src/ # Modular Python scripts
│ ├── Data_Preprocessing.py # Data cleaning, encoding, and balancing
│ ├── Modeling_SVM.py # SVM baseline and tuned model
│ ├── Modeling_NN.py # Neural network baseline and tuned model
│ ├── Modeling_XGBoost.py # XGBoost baseline and tuned model
│ ├── Interpretation_of_Variables.py # SHAP and permutation importance
│ └── init.py
│
├── data/
│ └── bank_churn.xlsx # Public dataset
│
├── requirements_main.txt
├── requirements_pycaret.txt
├── .gitignore
└── README.md
---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing  
- Encoded binary and categorical variables using `get_dummies()`.  
- Removed irrelevant features such as `clientnum`.  
- Split data into training and test sets (80/20).  
- Applied **SMOTE-Tomek balancing** to handle class imbalance.  

### 2️⃣ Modeling  

#### 🔹 Support Vector Machine (SVM)
- Compared baseline and tuned models using **GridSearchCV**.  
- The tuned version reduced false negatives from 108 → 93 and increased true positives from 212 → 227.  
- Improved identification of churned clients.

#### 🔹 Neural Network (TensorFlow/Keras)
- Implemented both baseline and tuned architectures.  
- The tuned network achieved significant improvements:  
  - False negatives reduced from 303 → 76  
  - True positives increased from 17 → 244  
- Outperformed SVM in AUC and overall accuracy.

#### 🔹 XGBoost
- Developed baseline and tuned models using **RandomizedSearchCV**.  
- The tuned model achieved the **best overall performance**, with:  
  - Fewer false negatives (42 → 37)  
  - Higher true positives (278 → 283)  
  - AUC score ≈ **0.99**  
- Selected as the **best-performing model**.

---

## 📊 Model Interpretation

### 🔸 Permutation Importance
- **Most influential features:**  
  - `total_trans_ct`  
  - `total_trans_amt`  
  - `total_revolving_bal`  
- These variables showed the largest impact on AUC, reducing it by up to 0.16 points when perturbed.

### 🔸 SHAP Values
- **Features increasing churn probability:**  
  - High `total_trans_ct` and `total_trans_amt` → intense activity before leaving  
  - High `total_ct_chng_q4_q1` and long `months_inactive_12_mon` → abrupt behavioral changes  
  - `marital_status_Single` → demographic correlation with higher churn  

- **Features decreasing churn probability:**  
  - High `total_revolving_bal` → financial engagement reduces churn risk  
  - High `contacts_count_12_mon` → frequent interactions improve retention  
  - Higher education levels (Graduate, Post-Graduate) → more stable relationships  

---

## 🧠 Key Findings

| Model | AUC | Strengths |
|--------|-----|-----------|
| **SVM (tuned)** | 0.96 | Balanced precision, interpretable |
| **Neural Network (tuned)** | 0.98 | High recall, captures nonlinear patterns |
| **XGBoost (tuned)** | **0.99** | Best trade-off, minimal false negatives |

- The **XGBoost tuned model** achieved the best performance, minimizing false negatives (clients incorrectly classified as non-churners).  
- The **transactional behavior** features (`total_trans_ct`, `total_trans_amt`) were the most decisive predictors across all models.

---

## 🧰 Technologies Used

| Category | Libraries |
|-----------|------------|
| Data Manipulation | Pandas, NumPy |
| Modeling | Scikit-learn, TensorFlow/Keras, XGBoost |
| Sampling | imbalanced-learn (SMOTE-Tomek) |
| Visualization | Matplotlib, Seaborn |
| Model Interpretation | SHAP, Permutation Importance |
| Environment | Python 3.10, Virtual Envs (`venv`, `pycaret_env`) |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/camilodel02/ChurnPrediction_ML_Framework.git
   cd ChurnPrediction_ML_Framework
👨‍💻 Author

Camilo Delgado Burbano
Pontificia Universidad Javeriana — Industrial Engineering
