# 🧠 Bank Customer Churn Prediction Framework 📊  
**Final Project — Analytics Methods and Aplications  II (Pontificia Universidad Javeriana)**  

This project develops a complete **Machine Learning framework** to predict customer churn in a retail bank --> **[Notebook_Main.ipynb](./Notebook_Main.ipynb)**. 
It integrates data preprocessing, model training, hyperparameter tuning, and interpretability analysis using multiple algorithms (SVM, Neural Networks, XGBoost, and PyCaret).  
The goal is to identify clients most likely to leave the bank and understand the behavioral and financial factors driving churn.

---

## 🧩 Project Overview

A financial institution experienced a **15% client reduction in the last quarter**, prompting an analysis to predict which customers are likely to churn.  
This repository provides a modular and explainable pipeline to classify customers into churn (1) or retain (0), enabling proactive retention actions.

---

## 🗂️ Repository Structure

```text
ChurnPrediction_ML_Framework/
│
├── Notebook_Main.ipynb           # Manual ML framework (SVM, NN, XGBoost)
├── Pycaret_Notebook.ipynb        # Automated modeling with PyCaret
│
├── src/                          # Modular Python scripts
│   ├── Data_Preprocessing.py     # Cleaning, encoding, balancing (SMOTE-Tomek)
│   ├── Modeling_SVM.py           # SVM baseline + tuned + ROC + evaluation
│   ├── Modeling_NN.py            # NN baseline + tuned + ROC + SHAP helpers
│   ├── Modeling_XGBoost.py       # XGBoost baseline + tuned + ROC + importance
│   ├── Interpretation_of_Variables.py # Permutation importance + SHAP for SVM
│   └── __init__.py
│
├── data/
│   └── bank_churn.xlsx           # Public/anonymized dataset
│
├── requirements_main.txt
├── requirements_pycaret.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Methodology

### 🟦 Data Preprocessing  
- Encoded binary and categorical variables using `get_dummies()`.  
- Removed irrelevant features such as `clientnum`.  
- Split data into training and test sets (80/20).  
- Applied **SMOTE-Tomek balancing** to handle class imbalance.  

---

### 🟩 Modeling (Manual Framework → Notebook_Main)
Baseline and tuned models were implemented in this section.

#### 🔹 Support Vector Machine (SVM)
- Compared baseline and tuned models using **GridSearchCV**.  
- The tuned version reduced false negatives from **108 → 93** and increased true positives from **212 → 227**.  
- Improved identification of churned clients.

#### 🔹 Neural Network (TensorFlow/Keras)
- Implemented both baseline and tuned architectures.  
- The tuned network achieved significant improvements:  
  - False negatives reduced from **303 → 76**  
  - True positives increased from **17 → 244**  
- Outperformed SVM in AUC and overall accuracy.

#### 🔹 XGBoost
- Developed baseline and tuned models using **RandomizedSearchCV**.  
- The tuned model achieved the **best overall performance**, with:  
  - Fewer false negatives (**42 → 37**)  
  - Higher true positives (**278 → 283**)  
  - **AUC ≈ 0.99**  
- Selected as the **best-performing model**.

---

### 🟨 PyCaret (AutoML Validation → Pycaret_Notebook)
To validate the robustness of the manual framework, an AutoML experiment was conducted using **PyCaret**.

#### 🔸 Findings
- Automatically compared multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, etc.).  
- **Light Gradient Boosting Machine** was selected as the best-performing model.  
- Generated performance plots:  
  - **Confusion Matrix** → High recall for churn class.  
  - **ROC Curve** → Excellent AUC (~0.99).  
  - **Feature Importance** → Consistent with manual findings (`total_trans_ct`, `total_trans_amt`, `total_revolving_bal`).  
  - **Class Report** → Balanced precision–recall trade-off.  
- The **PyCaret Dashboard** provided interactive evaluation and explainability insights.  

#### 🔸 Interpretation
- PyCaret showed that **Light Gradient Boosting Machine** consistently outperformed other models.  
- The same top features emerged as key churn drivers, validating the interpretability and consistency of the full ML pipeline.

---

## 📊 Model Interpretation

### 🔸 Permutation Importance
- **Most influential features:**  
  - `total_trans_ct`  
  - `total_trans_amt`  
  - `total_revolving_bal`  
- These variables showed the largest impact on AUC, reducing it by up to **~0.16** when perturbed.

### 🔸 SHAP Values
- **Features increasing churn probability:**  
  - High `total_trans_ct` and `total_trans_amt` → intense activity before leaving.  
  - High `total_ct_chng_q4_q1` and long `months_inactive_12_mon` → abrupt behavioral changes.  
  - `marital_status_Single` → demographic correlation with higher churn.  

- **Features decreasing churn probability:**  
  - High `total_revolving_bal` → financial engagement reduces churn risk.  
  - High `contacts_count_12_mon` → frequent interactions improve retention.  
  - Higher education levels (Graduate, Post-Graduate) → more stable relationships.  

---

## 🧠 Key Findings

| Model | AUC | Strengths |
|------|-----|-----------|
| **SVM (tuned)** | ~0.96 | Balanced precision; interpretable |
| **Neural Network (tuned)** | ~0.98 | High recall; captures nonlinear patterns |
| **XGBoost (tuned)** | **~0.99** | Best trade-off; minimal false negatives |
| **PyCaret (AutoML)** | ~0.99 | Light Gradient Boosting Machine was the best model |

- The **XGBoost tuned model** achieved the best performance, minimizing **false negatives** (clients incorrectly classified as non-churners) and maximizing the **true positives** (clients correctly classified as churners.  
- **Transactional behavior** features (`total_trans_ct`, `total_trans_amt`) are the most decisive predictors across all approaches.  
- PyCaret validates the robustness and reproducibility of boosting models.

---

## 🧰 Technologies Used

| Category | Libraries |
|---------|-----------|
| Data Manipulation | Pandas, NumPy |
| Modeling | Scikit-learn, TensorFlow/Keras, XGBoost, PyCaret |
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
   ```

2. **Use Python 3.10.11 kernel**

   Make sure you are running the **[Notebook_Main.ipynb](./Notebook_Main.ipynb)** with a Python 3.10.11 interpreter (the same version used to develop the project).

   ```bash
   # Verify your Python version
   python --version
   # Expected output:
   # Python 3.10.11

   # Then install the required dependencies
   pip install -r requirements_main.txt

3. **PyCaret environment**

   Make sure you are running the **[Pycaret_Notebook.ipynb](./Pycaret_Notebook.ipynb)** with a virtual env (3.10.11).

   ```bash
   python -m venv pycaret_env
   .\pycaret_env\Scripts\activate
   pip install -r requirements_pycaret.txt
   ```

5. **Run the notebooks with their respective envs**
   - `Notebook_Main.ipynb` → Manual framework (SVM, NN, XGBoost, SHAP, Permutation Importance).  
   - `Pycaret_Notebook.ipynb` → Automated benchmark with PyCaret.  

---

## 👨‍💻 Authors

**Camilo Delgado Burbano - Cristion Ivan Pulido Molano**  
Pontificia Universidad Javeriana — Industrial Engineering  
📧 [camilodelgadoburbano@gmail.com](mailto:camilodelgadoburbano@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/camilodelgadoburbano)

---

## 🏁 License

This project is for **academic and educational purposes**.  
You are welcome to use or adapt the code for learning and research.
