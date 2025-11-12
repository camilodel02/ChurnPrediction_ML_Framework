
''' This module contains functions for interpreting variable importance
    in machine learning models, including permutation importance and shap values. '''

import pandas as pd
import numpy as np
import shap
import re
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from src.Modeling_NN import KerasProbaWrapper
def permutation_imp(model, X, y, *, scaler=None, top_k=15, show=True):
    """
    Permutation importance que funciona tanto para sklearn (con predict_proba)
    como para Keras (usando el wrapper).
    """
    est = model if hasattr(model, "predict_proba") else KerasProbaWrapper(model, scaler, X.columns)

    r = permutation_importance(
        est, X, y,
        n_repeats=20,
        random_state=472,
        scoring="roc_auc"   
    )

    means = r.importances_mean
    order = np.argsort(-means)[:top_k]
    names = np.array(getattr(X, "columns", np.arange(X.shape[1])))[order]
    vals = means[order]

    if show:
        plt.figure(figsize=(7, max(3, top_k * 0.4)))
        plt.barh(range(len(vals)), vals)
        plt.gca().invert_yaxis()
        plt.yticks(range(len(vals)), names)
        plt.xlabel("Permutation importance (Δ AUC)")
        plt.title(f"Top features by permutation importance – {est.__class__.__name__}")
        plt.tight_layout()
        plt.show()

def partial_dependence_plots(model, X, features, *, scaler=None, show=True):
    """
    Grafica Partial Dependence Plots para modelos de Keras o sklearn.
    - model: modelo entrenado (Keras o scikit-learn)
    - X: matriz de features
    - features: lista de features (nombres o índices) para graficar
    - scaler: StandardScaler usado en entrenamiento (o None si no aplica)
    """
    if scaler is not None:
        X_sc = scaler.transform(X)
    else:
        X_sc = X

    est = model if hasattr(model, "predict_proba") else KerasProbaWrapper(model, scaler)

    display = PartialDependenceDisplay.from_estimator(
        est, X_sc, features, kind="average", grid_resolution=50
    )

    if show:
        plt.show()
    else:
        plt.close()

def shap_for_svm(svm_model, X_train, X_test, scaler=None):
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
        X_test_df  = pd.DataFrame(X_test,  columns=[f"f{i}" for i in range(X_test.shape[1])])
    else:
        X_train_df, X_test_df = X_train.copy(), X_test.copy()

    # Background (datos de referencia)
    bg = shap.sample(X_train, 50, random_state=472)
    xt = shap.sample(X_test_df, 50, random_state=472)

    # Función de predicción (probabilidad de la clase positiva)
    def predict_p1(X):
        X = pd.DataFrame(X, columns=X_train_df.columns)
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        if hasattr(svm_model, "predict_proba"):
            return svm_model.predict_proba(X)[:, 1]
        else:
            # Si no hay proba, usa decision_function y calibra antes en entrenamiento
            # (o convierte con logística solo si el SVM fue calibrado)
            return svm_model.decision_function(X)  

    explainer = shap.KernelExplainer(predict_p1, bg)
    sv = explainer.shap_values(xt)   # sv: (n_samples, n_features)

    # Gráficos
    #shap.summary_plot(sv, xt, plot_type="bar", show=True)
    shap.summary_plot(sv, xt, show=True)







