import os
import warnings

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             make_scorer, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, r2_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from tabulate import tabulate


def analyse_univariee(df):
    
    """
    Effectue une analyse univariée sur toutes les variables d'un DataFrame.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés. 
    Pour les variables non numériques, seules des statistiques descriptives sont générées.

    Paramètres
    ----------
    df : DataFrame
        DataFrame pandas à analyser.
    """
    
    for col in df.columns:
        # Affichage du nom de la variable analysée
        print(f"Analyse univariée pour la variable '{col}':")
        # Vérifie si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Si oui, calcul des statistiques descriptives
            analyse = df[col].describe().to_frame().transpose()
            analyse['skew'] = df[col].skew()
            analyse['kurtosis'] = df[col].kurt()
        else:
            # Si non, calcul des statistiques de base pour les variables catégorielles
            analyse = df[col].describe(include=['O']).to_frame().transpose()

        # Affichage du résultats sous forme de tableau
        print(tabulate(analyse, headers='keys', tablefmt='fancy_grid'))

        # Vérifie si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Si oui, création d'une figure pour tracer les graphiques
            plt.figure(figsize=(7,4))

            # Trace un histogramme avec une courbe de densité de probabilité
            ax1 = sns.histplot(data=df, x=col, kde=True)
            ax1.set(title=f"Analyse univariée de la variable {col}")
            if ax1.lines:
                ax1.lines[0].set(color="orange", linewidth=3)
            # Ajoute des lignes verticales pour la moyenne et la médiane
            ax1.axvline(df[col].mean(), color="crimson", linestyle="dotted", label=f"Moyenne {col}")    
            plt.axvline(df[col].median(), color="black", linestyle="dotted", label=f"Médiane {col}")
            plt.legend()
            plt.show()

            # Création d'une autre figure pour le boxplot
            plt.figure(figsize=(7,4))

            # Trace un boxplot pour visualiser la répartition des données
            ax2 = sns.boxplot(x=col, data=df, linewidth=3, color="white", showmeans=True)
            # Donner un titre dynamique au boxplot
            ax2.set(title=f"Répartition de la variable {col}")
            plt.show()
        
        # Insértion d'une ligne vide pour séparer les résultats de chaque variable
        print("\n")
        
        
def comp_analyse_univariee(df1, df2):
    
    """
    Effectue une analyse univariée sur toutes les variables de deux DataFrames.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés. 
    Pour les variables non numériques, seules des statistiques descriptives sont générées.

    Paramètres
    ----------
    df1, df2 : DataFrame
        DataFrames pandas à analyser.
    """
    
    for col in df1.columns:
        # Assurer que la colonne existe dans les deux DataFrames
        if col not in df2.columns:
            continue
            
        # Affichage du nom de la variable analysée
        print(f"Analyse univariée pour la variable '{col}':")
        
        # Vérifie si la colonne est numérique
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(df2[col].dtype, np.number):
            # Si oui, calcul des statistiques descriptives
            analyse1 = df1[col].describe().to_frame().transpose()
            analyse1['skew'] = df1[col].skew()
            analyse1['kurtosis'] = df1[col].kurt()

            analyse2 = df2[col].describe().to_frame().transpose()
            analyse2['skew'] = df2[col].skew()
            analyse2['kurtosis'] = df2[col].kurt()

        else:
            # Si non, calcul des statistiques de base pour les variables catégorielles
            analyse1 = df1[col].describe(include=['O']).to_frame().transpose()
            analyse2 = df2[col].describe(include=['O']).to_frame().transpose()

        # Affichage du résultats sous forme de tableau
        print("DataFrame 1")
        print(tabulate(analyse1, headers='keys', tablefmt='fancy_grid'))
        print("DataFrame 2")
        print(tabulate(analyse2, headers='keys', tablefmt='fancy_grid'))

        # Vérifie si la colonne est numérique
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(df2[col].dtype, np.number):
            # Si oui, création d'une figure pour tracer les graphiques
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Trace un histogramme avec une courbe de densité de probabilité
            sns.histplot(data=df1, x=col, kde=True, ax=axes[0])
            axes[0].set(title=f"Analyse univariée de la variable {col} (DataFrame 1)")

            sns.histplot(data=df2, x=col, kde=True, ax=axes[1])
            axes[1].set(title=f"Analyse univariée de la variable {col} (DataFrame 2)")

            plt.show()

            # Création d'une autre figure pour le boxplot
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Trace un boxplot pour visualiser la répartition des données
            sns.boxplot(x=col, data=df1, linewidth=3, color="white", showmeans=True, ax=axes[0])
            axes[0].set(title=f"Répartition de la variable {col} (DataFrame 1)")

            sns.boxplot(x=col, data=df2, linewidth=3, color="white", showmeans=True, ax=axes[1])
            axes[1].set(title=f"Répartition de la variable {col} (DataFrame 2)")

            plt.show()
        
        # Insértion d'une ligne vide pour séparer les résultats de chaque variable
        print("\n")


 

def evaluation_model(model, kfold, X, y):
    """
    Fonction pour calculer et afficher les indicateurs d'évaluation d'un modèle de régression.

    Paramètres:
    model (sklearn model) : Le modèle à évaluer.
    kfold (KFold object) : L'objet KFold pour la validation croisée.
    X (DataFrame) : Les caractéristiques à utiliser pour l'entraînement et la validation.
    y (Series) : La cible à prédire.

    Renvoie:
    Aucun. Affiche les indicateurs d'entraînement et de validation.
    """

    # Initialisation des listes pour les indicateurs
    r2_scores_train, r2_scores_val = [], []
    mae_scores_train, mae_scores_val = [], []
    mse_scores_train, mse_scores_val = [], []
    mape_scores_train, mape_scores_val = [], []

    # Fonction interne pour calculer le MAPE
    def calculer_mape(y_reel, y_pred): 
        return np.mean(np.abs((y_reel - y_pred) / y_reel)) * 100

    # Boucle de validation croisée
    for indice_entrainement, indice_validation in kfold.split(X):
        X_entrainement, X_validation = X.iloc[indice_entrainement], X.iloc[indice_validation]
        y_entrainement, y_validation = y.iloc[indice_entrainement], y.iloc[indice_validation]
    
        # Prédiction et évaluation du modèle sur l'ensemble d'entraînement
        predictions_entrainement = model.predict(X_entrainement)
        r2_scores_train.append(r2_score(y_entrainement, predictions_entrainement))
        mae_scores_train.append(mean_absolute_error(y_entrainement, predictions_entrainement))
        mse_scores_train.append(mean_squared_error(y_entrainement, predictions_entrainement))
        mape_scores_train.append(calculer_mape(y_entrainement, predictions_entrainement))
    
        # Prédiction et évaluation du modèle sur l'ensemble de validation
        predictions_validation = model.predict(X_validation)
        r2_scores_val.append(r2_score(y_validation, predictions_validation))
        mae_scores_val.append(mean_absolute_error(y_validation, predictions_validation))
        mse_scores_val.append(mean_squared_error(y_validation, predictions_validation))
        mape_scores_val.append(calculer_mape(y_validation, predictions_validation))
    
    # Affichage des indicateurs d'entraînement
    print("indicateurs d'entraînement\nScore R^2 moyen :", np.mean(r2_scores_train), 
          "\nMAE moyen :", np.mean(mae_scores_train), 
          "\nMSE moyen :", np.mean(mse_scores_train),
          "\nMAPE moyen :", np.mean(mape_scores_train))

    # Affichage des indicateurs de validation
    print("\nindicateurs de validation\nScore R^2 moyen :", np.mean(r2_scores_val), 
          "\nMAE moyen :", np.mean(mae_scores_val),
          "\nMSE moyen :", np.mean(mse_scores_val),
          "\nMAPE moyen :", np.mean(mape_scores_val))
