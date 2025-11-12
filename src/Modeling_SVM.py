

''' This module contains functions to build, train, and evaluate SVM models,
    including hyperparameter tuning and plotting ROC curves. '''


import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
from src.Data_Preprocessing import preprocessing, balance_data



def svm_model(X_train, Y_train, X_test, Y_test):
    ''' Trains and evaluates an SVM classifier. Returns the trained model and evaluation metrics. 
        Additionally, it balances the training data using the specified method before training. '''
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("Training SVM Classifier...\n")

    model = svm.SVC(probability=True, random_state=472)
    model.fit(X_train_sc, Y_train)
    Y_pred = model.predict(X_test_sc)
    Y_proba = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        'accuracy': accuracy_score(Y_test, Y_pred),
        'roc_auc': roc_auc_score(Y_test, Y_proba),
        'precision': precision_score(Y_test, Y_pred),
        'recall': recall_score(Y_test, Y_pred),
        'f1_score': f1_score(Y_test, Y_pred),
        'confusion_matrix': confusion_matrix(Y_test, Y_pred)
    }
    metrics_report = classification_report(Y_test, Y_pred)
    print("SVM Classifier metrics report:\n", metrics_report,"\n")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    #print ("Metrics:\n", metrics)
    return model, X_test_sc, Y_test


def best_svm(X_train, y_train):
    """
    Hace GridSearchCV sobre un pipeline [Scaler -> SVC].
    Devuelve:
      - best_model: pipeline ya entrenado (incluye scaler y SVC)
      - best_params: dict con los mejores hiperparámetros
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", svm.SVC(probability=True, random_state=472))
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf", "poly"],
        "svc__gamma": ["scale", "auto"],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",  
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model  = grid.best_estimator_   # ← pipeline ya fit
    best_params = grid.best_params_
    print("Best hyperparameters found:", best_params)
    return best_model, best_params


def curva_roc(model, X_test_sc, Y_test, save_path=None, show=True):
    """Plots and optionally saves the ROC curve."""
    disp = RocCurveDisplay.from_estimator(model, X_test_sc, Y_test)
    plt.title('ROC Curve SVM')
    
    # Guarda la imagen si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve SVM saved at: {save_path}")
    
    # Muestra la figura en el notebook
    if show:
        plt.show()
    
    # Cierra la figura para evitar duplicados si llamas muchas veces
    plt.close()


def evaluate_svm(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(Y_test, Y_pred),
        'roc_auc': roc_auc_score(Y_test, Y_proba),
        'precision': precision_score(Y_test, Y_pred),
        'recall': recall_score(Y_test, Y_pred),
        'f1_score': f1_score(Y_test, Y_pred),
        'confusion_matrix': confusion_matrix(Y_test, Y_pred)
    }

    report = classification_report(Y_test, Y_pred)
    print("SVM Classifier metrics report:\n", report,"\n")
    print("Confusion Matrix:\n", metrics['confusion_matrix'])


    return metrics, report

# ------------Example usage: ------------

#df = pd.read_excel(r"C:\Users\LENOVO\Documents\D\Documentos\Javeriana\IX semestre\Analítica 2\Proyecto_Final\data\bank_churn.xlsx")
#print("Reading data completed. \n")
#print("Starting preprocessing... \n")
#x, y = preprocessing(df)
#print("Preprocessing completed. \n")
#print("Data balancing completed. \n")
#print("Running SVM Classifier ...")
#model_vf, X_test_sc, Y_test = svm_model(x, y, test_size=0.2)
#
#curva_roc(model_vf, X_test_sc, Y_test, show=True)
