
''' This module contains functions to build, train, and evaluate Neural Network models,
    including hyperparameter tuning, shap and plotting ROC curves. '''

import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.base import BaseEstimator, ClassifierMixin


# ------------------------------
# -----------BASELINE-----------
# ------------------------------
def build_nn(input_dim, hidden_units=[128,64], dropout=0.2, l2_reg=1e-4, lr=1e-3):
    reg = regularizers.l2(l2_reg)
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for h in hidden_units:
        model.add(layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def train_nn_baseline(X_train, y_train, X_test, y_test,
                      hidden_units=[128,64], dropout=0.2,
                      l2_reg=1e-4, lr=1e-3, epochs=60, batch_size=256):
    
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = build_nn(X_train_sc.shape[1], hidden_units, dropout, l2_reg, lr)

    es = callbacks.EarlyStopping(monitor="val_auc", patience=8,
                                 restore_best_weights=True, mode="max")

    model.fit(X_train_sc, y_train,
              validation_split=0.2,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[es],
              verbose=0)

    y_proba = model.predict(X_test_sc).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    
    return model, scaler, metrics

# -------------------------------------------
# -----------HYPERPARAMETER TUNING-----------
# -------------------------------------------

def train_nn_tuned(X_train, y_train, X_test, y_test,
                   hidden_grid=[[128,64],[64,32],[128,64,32]],
                   dropout_grid=[0.0,0.2],
                   l2_grid=[0.0,1e-4],
                   lr_grid=[1e-3,5e-4],
                   batch_grid=[128],
                   epochs=60):

    # split interno (train → train/val)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=472)

    scaler = StandardScaler().fit(X_tr)
    X_tr_sc = scaler.transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    best_auc = -1
    best_model = None
    best_params = None

    print("Starting hyperparameter tuning for Neural Network...\n")
    for h, dr, l2v, lr, bs in product(hidden_grid, dropout_grid, l2_grid, lr_grid, batch_grid):

        tf.keras.backend.clear_session()
        model = build_nn(X_tr_sc.shape[1], h, dr, l2v, lr)

        es = callbacks.EarlyStopping(
            monitor="val_auc", patience=8,
            restore_best_weights=True, mode="max")

        model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_val_sc, y_val),
            epochs=epochs,
            batch_size=bs,
            verbose=0,
            callbacks=[es]
        )

        val_proba = model.predict(X_val_sc).ravel()
        val_auc = roc_auc_score(y_val, val_proba)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_params = {"hidden": h, "dropout": dr, "l2": l2v, "lr": lr, "batch": bs}

    # evaluar en TEST
    y_proba = best_model.predict(X_test_sc, verbose=0 ).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return best_model, scaler, metrics, best_params


# ------------------------------
# -----------GRAPHS-------------
# ------------------------------

def plot_roc_curve(model, scaler, X_test, y_test, title="ROC Curve - Neural Network",
                   save_path=None, show=True):
    """
    Grafica la curva ROC para modelos de TensorFlow o scikit-learn.
    
    - model: modelo entrenado (Keras o scikit-learn)
    - scaler: StandardScaler usado en entrenamiento (o None si no aplica)
    - X_test: matriz de test
    - y_test: etiquetas
    """
    if scaler is not None:
        X_test_sc = scaler.transform(X_test)
    else:
        X_test_sc = X_test

    # Probabilidades (NN usa .predict, SVM usa .predict_proba)
    if hasattr(model, "predict_proba"):   # SVM, RF, etc.
        y_score = model.predict_proba(X_test_sc)[:,1]
    else:  # TensorFlow
        y_score = model.predict(X_test_sc, verbose = 0).ravel()

    disp = RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

# ------------------------------ Class KerasProbaWrapper ------------------------------
class KerasProbaWrapper:
    """Wrapper para usar modelos Keras en APIs de sklearn que requieren
       .predict_proba(), como PermutationImportance o PartialDependenceDisplay.
       Usa un StandardScaler para transformar los datos antes de predecir."""
    
    _estimator_type = "classifier"

    def __init__(self, model, scaler, cols):
        self.model = model
        self.scaler = scaler
        self.cols = list(cols)
        # Para APIs que miran classes_
        self.classes_ = np.array([0, 1], dtype=int)

    # Cumplir interfaz sklearn (no entrena)
    def fit(self, X, y=None):
        return self

    # Necesario para compatibilidad con get_params/set_params
    def get_params(self, deep=True):
        return {"model": self.model, "scaler": self.scaler, "cols": self.cols}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.cols)
        else:
            X = X[self.cols]
        Xs = self.scaler.transform(X)
        p1 = self.model.predict(Xs, verbose=0).ravel()
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        p = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, p)
    
# ------------------------------
# -----------SHAP---------------
# ------------------------------

def compute_shap_forceplot(
    model,scaler,X_train,X_test,
    wrapper_cls,                # e.g., KerasProbaWrapper
    class_index=1,              # clase positiva por defecto (binaria)
    bg_size=100,
    show_size=100,
    nsamples=100,
    random_state=0,
    save_html=None,             # p.ej. "shap_force.html" para guardar
    return_plot=False           # si True, retorna el objeto plot (útil en HTML)
):
    """
    Calcula valores SHAP con KernelExplainer y genera force plot grupal.

    Params
    ------
    model, scaler : tu red y escalador entrenados
    X_train, X_test : pandas DataFrames con mismas columnas
    wrapper_cls : clase que expone .predict_proba(X) usando model+scaler
    class_index : índice de clase a explicar (1 = positiva en binario)
    bg_size : tamaño del background para el explainer
    show_size : número de muestras a explicar/mostrar
    nsamples : muestras internas de KernelSHAP (trade-off precisión/tiempo)
    save_html : ruta para guardar el force plot en HTML (opcional)
    return_plot : si True, retorna el objeto del plot además de sv/exp/X_show

    Returns
    -------
    sv : np.ndarray shape (n_samples, n_features)  # SHAP de la clase elegida
    exp : float                                    # expected value de esa clase
    X_show : DataFrame                             # subconjunto explicado
    plot_obj : (opcional) objeto retornado por shap.force_plot
    """

    # 1) Wrapper y columnas
    cols = list(scaler.feature_names_in_)
    wrapper = wrapper_cls(model, scaler, cols)

    # 2) Muestras de background y a mostrar (manteniendo columnas)
    X_bg   = shap.utils.sample(X_train, bg_size, random_state=random_state)
    X_show = shap.utils.sample(X_test,  show_size, random_state=random_state)

    # 3) Explainer + valores SHAP
    kernel_shap = shap.KernelExplainer(wrapper.predict_proba, X_bg)
    shap_values = kernel_shap.shap_values(X_show, nsamples=nsamples)

    # 4) Selección robusta de la clase objetivo y normalización de forma
    if isinstance(shap_values, list):
        # lista por clase
        idx = min(class_index, len(shap_values)-1)
        sv = shap_values[idx]
    else:
        sv = shap_values
        # intentar descifrar ejes si vienen con eje de clase
        if getattr(sv, "ndim", 0) == 3:
            # (n_samples, n_features, n_classes)
            if sv.shape[2] > 1:
                idx = min(class_index, sv.shape[2]-1)
                sv = sv[:, :, idx]
            # (n_classes, n_samples, n_features)
            elif sv.shape[0] > 1:
                idx = min(class_index, sv.shape[0]-1)
                sv = sv[idx]
            # (n_samples, n_classes, n_features)
            elif sv.shape[1] > 1:
                idx = min(class_index, sv.shape[1]-1)
                sv = sv[:, idx, :]

    # asegurar (n_samples, n_features)
    if sv.shape[0] != X_show.shape[0] and sv.shape[1] == X_show.shape[0]:
        sv = sv.T

    # 5) Expected value de la clase elegida
    exp_val = kernel_shap.expected_value
    if isinstance(exp_val, (list, tuple, np.ndarray)):
        idx = min(class_index, len(exp_val)-1)
        exp = exp_val[idx]
    else:
        exp = exp_val

    # 6) Force plot (opcionalmente guardar a HTML)
    plot_obj = shap.force_plot(exp, sv, X_show)
    if save_html is not None:
        shap.save_html(save_html, plot_obj)

    if return_plot:
        return sv, exp, X_show, plot_obj
    return sv, exp, X_show

