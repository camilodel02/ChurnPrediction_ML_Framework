
''' This module contains functions to build, train, and evaluate XGBoost models,
    including hyperparameter tuning, variable importance and plotting ROC curves. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def xgboost_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)

    print("XGBoost Classifier metrics report:\n", report, "\n")
    print("Confusion Matrix:\n", cm)
    return model 

def xgboost_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    from sklearn.model_selection import RandomizedSearchCV
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    rs = RandomizedSearchCV(xgb, param_distributions=param_dist,
                            n_iter=10, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
    rs.fit(X_train, y_train)
    best_model  = rs.best_estimator_
    best_params = rs.best_params_

    y_pred = best_model.predict(X_test)
    print("XGBoost tuned best hyperparameters found:", best_params, "\n")
    print("XGBoost tuned Classifier metrics report:\n", classification_report(y_test, y_pred), "\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return best_model, best_params

def roc_curve_xgboost(model, X_test, y_test, show=True):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - XGBoost')
    plt.legend(loc='lower right')
    if show: plt.show()

def var_importance_xgboost(model, x_balanced, show=True):
    """
    Plots the feature importance of an XGBoost model.

    Parameters:
    model: Trained XGBoost model.
    X_train: Training feature set.
    show: Boolean to display the plot.

    Returns:
    None
    """
    var_importance = model.feature_importances_
    features = x_balanced.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': var_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=True).head(10)
    plt.figure(figsize=(10,6))
    plt.title('Top 10 Feature Importances - XGBoost')
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
